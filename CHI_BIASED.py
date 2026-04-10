#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import math
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
from scipy import stats
from scipy.stats import t as std_t
from scipy.stats import multivariate_t
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)
np.random.seed(0)
torch.manual_seed(0)


# DEFINITIONS


def _rqs_forward(x, widths, heights, derivatives, tail_bound):
    K = widths.shape[-1]
    W = torch.softmax(widths, dim=-1) * 2 * tail_bound
    H = torch.softmax(heights, dim=-1) * 2 * tail_bound
    D = torch.nn.functional.softplus(derivatives) + 1e-5
    cum_w = torch.cat([torch.full_like(W[..., :1], -tail_bound),
                       -tail_bound + torch.cumsum(W, dim=-1)], dim=-1)
    cum_h = torch.cat([torch.full_like(H[..., :1], -tail_bound),
                       -tail_bound + torch.cumsum(H, dim=-1)], dim=-1)
    inside = (x >= -tail_bound) & (x <= tail_bound)
    x_safe = x.clone()
    x_safe[~inside] = 0.0
    bin_idx = torch.searchsorted(cum_w[..., 1:-1].contiguous(),
                                 x_safe.unsqueeze(-1)).squeeze(-1).clamp(0, K - 1)
    def gather(t): return t.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    x_k, x_k1 = gather(cum_w[..., :-1]), gather(cum_w[..., 1:])
    y_k, y_k1 = gather(cum_h[..., :-1]), gather(cum_h[..., 1:])
    d_k, d_k1 = gather(D[..., :-1]), gather(D[..., 1:])
    s_k = gather(H) / gather(W)
    xi = ((x_safe - x_k) / (x_k1 - x_k + 1e-8)).clamp(0.0, 1.0)
    denom = s_k + (d_k + d_k1 - 2 * s_k) * xi * (1 - xi)
    z_inside = y_k + (y_k1 - y_k) * (s_k * xi**2 + d_k * xi * (1 - xi)) / (denom + 1e-8)
    log_jac_inside = (2 * torch.log(s_k + 1e-8)
                      + torch.log(d_k1 * xi**2 + 2 * s_k * xi * (1-xi) + d_k * (1-xi)**2 + 1e-8)
                      - 2 * torch.log(denom.abs() + 1e-8))
    z = torch.where(inside, z_inside, x)
    log_jac = torch.where(inside, log_jac_inside, torch.zeros_like(x))
    return z, log_jac

class RQSCouplingLayer(nn.Module):
    def __init__(self, num_bins=5, tail_bound=2.5):
        super().__init__()
        self.K, self.tail_bound = num_bins, tail_bound
        self.params = nn.Parameter(torch.randn(3 * num_bins + 1) * 0.01)
    def forward(self, x):
        N = x.shape[0]
        p = self.params.unsqueeze(0).expand(N, -1)
        z, lj = _rqs_forward(x.squeeze(-1), p[:, :self.K],
                              p[:, self.K:2*self.K], p[:, 2*self.K:], self.tail_bound)
        return z.unsqueeze(-1), lj

class AffineLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.shift = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return torch.exp(self.log_scale) * x + self.shift, self.log_scale.expand(x.shape[0])

class RQSFlow(nn.Module):
    def __init__(self, num_bins=5, tail_bound=2.5, depth=1):
        super().__init__()
        self.rqs_layers = nn.ModuleList([RQSCouplingLayer(num_bins, tail_bound) for _ in range(depth)])
        self.affine = AffineLayer()
    def forward(self, x):
        log_jac = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = x
        for layer in self.rqs_layers:
            z, lj = layer(z)
            log_jac = log_jac + lj
        z, lj = self.affine(z)
        return z, log_jac + lj

def ttf_inverse_torch(x, lam_pos, lam_neg, mu, sigma):
    y = (x - mu) / sigma
    s = torch.sign(y)
    s = torch.where(s == 0, torch.ones_like(s), s)
    lam_s = torch.where(s > 0, lam_pos * torch.ones_like(s), lam_neg * torch.ones_like(s))
    inner = torch.clamp(lam_s * y.abs() + 1.0, min=1e-30)
    erfc_val = torch.clamp(inner ** (-1.0 / lam_s), min=1e-30, max=2.0 - 1e-7)
    return s * (-torch.erfinv(1.0 - erfc_val)) * (2.0 ** 0.5)

def ttf_log_abs_jac_torch(z, lam_pos, lam_neg, sigma):
    s = torch.sign(z)
    s = torch.where(s == 0, torch.ones_like(s), s)
    lam_s = torch.where(s > 0, lam_pos * torch.ones_like(s), lam_neg * torch.ones_like(s))
    erfc_val = torch.clamp(torch.erfc(z.abs() / (2.0**0.5)), min=1e-30)
    return (torch.log(sigma) + (-lam_s - 1.0) * torch.log(erfc_val)
            + torch.log(torch.tensor(2.0 / (2.0 * math.pi)**0.5)) - 0.5 * z**2)

class TTFRQS(nn.Module):
    name = "ttf"
    def __init__(self, num_bins=5, tail_bound=2.5, depth=1):
        super().__init__()
        self.flow = RQSFlow(num_bins, tail_bound, depth)
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        lp_init = float(torch.distributions.Uniform(0.05, 1.0).sample([1]).item())
        ln_init = float(torch.distributions.Uniform(0.05, 1.0).sample([1]).item())
        self.log_lam_pos = nn.Parameter(torch.log(torch.tensor(lp_init)))
        self.log_lam_neg = nn.Parameter(torch.log(torch.tensor(ln_init)))
    @property
    def lam_pos(self): return torch.exp(self.log_lam_pos).clamp(0.02, 15.0)
    @property
    def lam_neg(self): return torch.exp(self.log_lam_neg).clamp(0.02, 15.0)
    def log_prob(self, x):
        sigma = torch.exp(self.log_sigma).clamp(1e-3)
        z_ttf = ttf_inverse_torch(x.squeeze(-1), self.lam_pos, self.lam_neg, self.mu, sigma)
        lj_ttf = -ttf_log_abs_jac_torch(z_ttf, self.lam_pos, self.lam_neg, sigma)
        z_base, lj_flow = self.flow(z_ttf.unsqueeze(-1))
        log_base = -0.5 * (z_base.squeeze(-1)**2 + torch.log(torch.tensor(2 * math.pi)))
        return log_base + lj_flow + lj_ttf
    def logpdf_np(self, x_np):
        self.eval()
        xt = torch.tensor(x_np, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            lp = self.log_prob(xt).numpy()
        return np.where(np.isfinite(lp), lp, -1e9)

def _train_model(model, x_trn, x_val, n_epochs=2000, lr=5e-3, batch_size=512,
                 early_stop_patience=100, eval_period=20, verbose=True, tag="model"):
    x_t = torch.tensor(x_trn, dtype=torch.float32).view(-1, 1)
    x_v = torch.tensor(x_val, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(x_t), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    best_val, patience, best_state, step = -np.inf, 0, None, 0
    data_iter = iter(loader)
    model.train()
    while step < n_epochs:
        try: (xb,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (xb,) = next(data_iter)
        opt.zero_grad()
        lp = model.log_prob(xb)
        loss = -lp[torch.isfinite(lp)].mean()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        step += 1
        if step % eval_period == 0:
            model.eval()
            with torch.no_grad():
                val_ll = model.log_prob(x_v)
                val_ll = val_ll[torch.isfinite(val_ll)].mean().item()
            model.train()
            if val_ll > best_val:
                best_val, patience = val_ll, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if verbose and step % (eval_period * 5) == 0:
                print(f"    [{tag}] step {step:5d}  val_ll={val_ll:.4f}")
            if patience >= early_stop_patience:
                if verbose: print(f"    [{tag}] early stop at step {step}")
                break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return model

class TTFMarginal:
    def __init__(self, model_cfg=None):
        self.model_cfg = model_cfg or dict(num_bins=5, tail_bound=2.5, depth=1)
        self._model = self._cdf_grid = self._x_grid = None
    def fit(self, x_np, n_steps=12000, lr=5e-3, batch_size=512,
            early_stop_patience=100, eval_period=20, verbose=True, tag="ttf-marginal"):
        x_np = np.asarray(x_np, dtype=np.float64)
        self._x_min, self._x_max = float(x_np.min()), float(x_np.max())
        self._model = TTFRQS(**self.model_cfg)
        _train_model(self._model, x_np, x_np, n_epochs=n_steps, lr=lr,
                     batch_size=batch_size, early_stop_patience=early_stop_patience,
                     eval_period=eval_period, verbose=verbose, tag=tag)
        self._x_grid = np.linspace(self._x_min - 0.1, self._x_max + 0.1, 4000)
        pdf_grid = np.exp(self._model.logpdf_np(self._x_grid))
        cdf_grid = np.concatenate([[0.0], np.cumsum(np.diff(self._x_grid) *
                                   0.5 * (pdf_grid[:-1] + pdf_grid[1:]))])
        self._cdf_grid = np.clip(cdf_grid / (cdf_grid[-1] + 1e-12), 0.0, 1.0)
        return self
    def predict(self, x_np, verbose=0):
        x_flat = np.asarray(x_np, dtype=np.float64).ravel()
        pdf = np.exp(self._model.logpdf_np(x_flat)).reshape(-1, 1).astype(np.float32)
        cdf = np.interp(x_flat, self._x_grid, self._cdf_grid).reshape(-1, 1).astype(np.float32)
        return [cdf, np.clip(pdf, 1e-9, None), np.clip(-pdf, 0.0, None)]
    def cdf(self, x_np): return self.predict(x_np)[0]


# GUMBEL COPULA


class GumbelCopula:
    def __init__(self, theta_max=30.0):
        self.theta = self.tau_hat = None
        self.theta_max = theta_max
    @staticmethod
    def _log_density(u, v, theta):
        u, v = np.clip(u, 1e-9, 1-1e-9), np.clip(v, 1e-9, 1-1e-9)
        lu, lv = -np.log(u), -np.log(v)
        A = lu**theta + lv**theta
        A1t = A ** (1.0 / theta)
        log_c = (-A1t + np.log(lu) + np.log(lv)
                 + (theta - 1.0) * (np.log(lu) + np.log(lv))
                 - (2.0 - 1.0 / theta) * np.log(A)
                 + np.log(A1t + theta - 1.0))
        return np.where(np.isfinite(log_c), log_c, -1e9)
    @staticmethod
    def _cdf(u, v, theta):
        u, v = np.clip(u, 1e-9, 1-1e-9), np.clip(v, 1e-9, 1-1e-9)
        return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta) ** (1.0/theta))
    @staticmethod
    def tau_from_theta(theta): return 1.0 - 1.0 / theta
    def fit(self, u, v, verbose=True):
        u, v = np.asarray(u, np.float64).ravel(), np.asarray(v, np.float64).ravel()
        self.tau_hat = float(stats.kendalltau(u, v)[0])
        result = minimize_scalar(
            lambda theta: -self._log_density(u, v, theta)[np.isfinite(
                self._log_density(u, v, theta))].mean(),
            bounds=(1.0 + 1e-6, self.theta_max), method="bounded")
        self.theta = float(result.x)
        if verbose:
            print(f"  Gumbel MLE theta={self.theta:.4f}  tau={self.tau_from_theta(self.theta):.4f}"
                  f"  lambda_U={self.upper_tail_dep:.4f}")
        return self
    @property
    def upper_tail_dep(self): return float(2.0 - 2.0 ** (1.0 / self.theta))
    def log_density(self, u, v): return self._log_density(u, v, self.theta)
    def cdf(self, u, v): return self._cdf(u, v, self.theta)
    def simulate(self, n, rng=None):
        if rng is None: rng = np.random.default_rng()
        alpha = 1.0 / self.theta
        if abs(alpha - 1.0) < 1e-6:
            return rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        phi = rng.uniform(-math.pi/2, math.pi/2, n)
        W = rng.exponential(1.0, n)
        phi0 = -math.pi / 2 * (1.0 - alpha) / alpha
        V = ((np.sin(alpha*(phi-phi0)) / (np.cos(phi)+1e-30)) ** (1.0/alpha)
             * (np.cos(phi - alpha*(phi-phi0)) / (W+1e-30)) ** ((1.0-alpha)/alpha))
        V = np.clip(V, 1e-10, None)
        e1, e2 = rng.exponential(1.0, n), rng.exponential(1.0, n)
        return (np.clip(np.exp(-(e1/V)**alpha), 1e-9, 1-1e-9),
                np.clip(np.exp(-(e2/V)**alpha), 1e-9, 1-1e-9))


# T-COPULA


class TCopula:
    def __init__(self): self.rho = self.nu = None
    def fit(self, u, v, verbose=True):
        u, v = np.clip(u, 1e-7, 1-1e-7), np.clip(v, 1e-7, 1-1e-7)
        rho_init = np.sin(np.pi / 2 * stats.kendalltau(u, v)[0])
        def neg_ll(params):
            rho_val, nu_val = params
            if not (-0.99 < rho_val < 0.99) or nu_val < 0.1: return 1e10
            x, y = std_t.ppf(u, df=nu_val), std_t.ppf(v, df=nu_val)
            cov = np.array([[1.0, rho_val], [rho_val, 1.0]])
            return -np.mean(multivariate_t.logpdf(np.column_stack([x,y]), shape=cov, df=nu_val)
                            - std_t.logpdf(x, df=nu_val) - std_t.logpdf(y, df=nu_val))
        res = minimize(neg_ll, x0=[rho_init, 4.0], bounds=[(-0.99, 0.99), (0.1, 50)])
        self.rho, self.nu = res.x
        if verbose:
            print(f"  t-Copula MLE rho={self.rho:.4f}  nu={self.nu:.4f}  lambda={self.tail_dep:.4f}")
        return self
    @property
    def tail_dep(self):
        return 2 * std_t.cdf(-np.sqrt((self.nu+1)*(1-self.rho)/(1+self.rho)), df=self.nu+1)
    def simulate(self, n, rng=None):
        if rng is None: rng = np.random.default_rng()
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        s = multivariate_t.rvs(shape=cov, df=self.nu, size=n, random_state=rng)
        return std_t.cdf(s[:, 0], df=self.nu), std_t.cdf(s[:, 1], df=self.nu)


# DATA


df_raw = pd.read_csv("wind_flood_loss_pairs_events2.0.csv")
df_clean = pd.DataFrame({
    "wind_loss":  pd.to_numeric(df_raw["wind_loss"],  errors="coerce"),
    "flood_loss": pd.to_numeric(df_raw["flood_loss"], errors="coerce")
}).dropna()
print(f"Loaded {len(df_clean):,} rows")

df_clean["log_wind"]  = np.log(df_clean["wind_loss"]  + 1)
df_clean["log_flood"] = np.log(df_clean["flood_loss"] + 1)
data = df_clean[["log_wind", "log_flood"]].values.astype(np.float32)
for i in range(2):
    data[:, i] = (data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min() + 1e-12)

X_train, X_test = train_test_split(data, test_size=0.33, random_state=42)
number_of_dimension = 2
data_domain = np.array([[X_train[:, i].min(), X_train[:, i].max()] for i in range(2)], dtype=np.float32)
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")


# FIT TTF MARGINALS


ttf_marginal_list = []
for i in range(number_of_dimension):
    print(f"\nFitting TTF marginal {i}...")
    m = TTFMarginal()
    m.fit(X_train[:, i].astype(np.float64), n_steps=30000, lr=5e-3, batch_size=512,
          early_stop_patience=100, eval_period=20, verbose=True, tag=f"ttf-dim{i}")
    ttf_marginal_list.append(m)

U_train = np.column_stack([ttf_marginal_list[i].cdf(X_train[:, i].reshape(-1,1)).ravel()
                           for i in range(number_of_dimension)]).astype(np.float64)
U_test  = np.column_stack([ttf_marginal_list[i].cdf(X_test[:,  i].reshape(-1,1)).ravel()
                           for i in range(number_of_dimension)]).astype(np.float64)

# FIT GUMBEL AND T-COPULAS


gumbel = GumbelCopula()
gumbel.fit(U_train[:, 0], U_train[:, 1])

t_cop = TCopula()
t_cop.fit(U_train[:, 0], U_train[:, 1])


# NEURAL COPULA


class _copula(keras.Model):
    def __init__(self):
        super().__init__()
        self.kernel_regularizer = tf.keras.regularizers.L2(l2=0.001)
        self.get_components = Lambda(lambda x: [x[:, i:i+1] for i in range(number_of_dimension)])
        self.cat_components = Lambda(lambda x: K.concatenate(x, axis=-1))
        self.dense_layer_list = [keras.layers.Dense(10, activation='tanh',
                                  kernel_regularizer=self.kernel_regularizer) for _ in range(5)]
        self.final_layer = keras.layers.Dense(1, activation='sigmoid',
                                              kernel_regularizer=self.kernel_regularizer)
        self.gradient_layers = Lambda(lambda x: K.gradients(x[0], x[1]))
        self.relu = keras.layers.ReLU()
    def call(self, inputs):
        components = self.get_components(inputs)
        x = self.cat_components(components)
        for layer in self.dense_layer_list: x = layer(x)
        cdf = self.final_layer(x)
        pdf = cdf
        for i in range(number_of_dimension):
            pdf = self.gradient_layers([pdf, components[i]])
        neg = self.relu(-pdf[0])
        return [cdf, 1e-9 + self.relu(pdf[0]), neg]

copula_model = _copula()

joint_log_cdfs_np = np.column_stack([
    ttf_marginal_list[i].cdf(X_train[:, i].reshape(-1,1)).ravel()
    for i in range(number_of_dimension)]).astype(np.float32)

# Boundary
number_boundary_points = 400
bd_data, bd_labels = [], []
for i in range(number_of_dimension):
    t = np.random.rand(number_boundary_points, number_of_dimension).astype(np.float32); t[:, i] = 0
    bd_data.append(t); bd_labels.append(t[:, i])
for i in range(number_of_dimension):
    t = np.ones([number_boundary_points, number_of_dimension], dtype=np.float32)
    t[:, i] = np.random.rand(number_boundary_points).astype(np.float32)
    bd_data.append(t); bd_labels.append(t[:, i])
jb_data   = np.expand_dims(np.concatenate(bd_data,   axis=0), axis=0)
jb_labels = np.expand_dims(np.concatenate(bd_labels, axis=0), axis=0)


number_partition_per_dim = 50
grids = np.meshgrid(*[np.linspace(0, 1, number_partition_per_dim) for _ in range(number_of_dimension)])
jn_data = np.expand_dims(np.column_stack([g.ravel() for g in grids]).astype(np.float32), axis=0)

# Log-likelihood
jl_data   = np.expand_dims(joint_log_cdfs_np, axis=0)
jl_labels = np.zeros([1, 1], dtype=np.float32) + 5.0

# Observation
rng_obs = np.random.default_rng(42)
number_observation_points = 400
n_bulk       = int(number_observation_points * 0.25)
n_mid_tail   = int(number_observation_points * 0.25)
n_upper_tail = int(number_observation_points * 0.35)
n_marg_tail  = number_observation_points - n_bulk - n_mid_tail - n_upper_tail

obs_bulk = rng_obs.random((n_bulk, number_of_dimension)).astype(np.float32)
obs_mid_tail = (0.7 + rng_obs.random((n_mid_tail, number_of_dimension)) * 0.295).astype(np.float32)
a = 5.0
raw = rng_obs.beta(a, 1, size=(n_upper_tail, number_of_dimension))
obs_upper_tail = (0.9 + raw * 0.099).astype(np.float32)
obs_marg_tail = rng_obs.random((n_marg_tail, number_of_dimension)).astype(np.float32)
tail_dim = rng_obs.integers(0, number_of_dimension, size=n_marg_tail)
raw_marg = rng_obs.beta(a, 1, size=n_marg_tail)
for idx, d in enumerate(tail_dim):
    obs_marg_tail[idx, d] = float(0.9 + raw_marg[idx] * 0.099)

obs_u = np.vstack([obs_bulk, obs_mid_tail, obs_upper_tail, obs_marg_tail])
jo_data   = np.expand_dims(obs_u, axis=0)
jo_labels = np.zeros([1, number_observation_points], dtype=np.float32)
for j, u in enumerate(obs_u):
    jo_labels[0, j] = np.all(joint_log_cdfs_np <= u, axis=1).mean()

# Build training model
jb_in = keras.layers.Input(shape=jb_data.shape[1:])
jn_in = keras.layers.Input(shape=jn_data.shape[1:])
jl_in = keras.layers.Input(shape=jl_data.shape[1:])
jo_in = keras.layers.Input(shape=jo_data.shape[1:])

jb_out = Lambda(lambda x: tf.expand_dims(x, axis=0), name="JB")(copula_model(jb_in[0])[0][:, 0])
_, jns_pdf, jns_neg = copula_model(jn_in[0])
jn_out = Lambda(lambda x: tf.reduce_sum(x, keepdims=True) / jn_in.shape[1], name="JN")(jns_neg)
js_out = Lambda(lambda x: tf.reduce_sum(x, keepdims=True) / (number_partition_per_dim-1)**2, name="JS")(jns_pdf)
jl_out = Lambda(lambda x: tf.reduce_sum(tf.math.log(x), keepdims=True) / jl_in.shape[1], name="JL")(copula_model(jl_in[0])[1])
jo_out = Lambda(lambda x: tf.expand_dims(x, axis=0), name="JO")(copula_model(jo_in[0])[0][:, 0])

joint_model_train = keras.Model(inputs=[jb_in, jn_in, jl_in, jo_in],
                                outputs=[jb_out, jn_out, js_out, jl_out, jo_out])
joint_model_train.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001),
                          loss="mae", loss_weights=[2, 1, 1, 1.0, 2])

print("\nTraining neural copula...")
joint_model_train.fit(
    x=[jb_data, jn_data, jl_data, jo_data],
    y=[jb_labels, np.zeros([1,1],dtype=np.float32), np.ones([1,1],dtype=np.float32),
       jl_labels, jo_labels],
    epochs=80000, verbose=0)
print("Training complete.")


# TAIL METRICS

def calculate_tail_metrics(u, v, thresholds):
    n = len(u)
    chi_list, chibar_list = [], []
    for q in thresholds:
        p_both = np.sum((u > q) & (v > q)) / n
        p_marg = 1 - q
        chi = p_both / p_marg if p_marg > 0 else 0
        chibar = (2 * np.log(p_marg) / np.log(p_both) - 1) if p_both > 0 and p_marg > 0 else -1
        chi_list.append(chi); chibar_list.append(chibar)
    return np.array(chi_list), np.array(chibar_list)

thresholds = np.linspace(0.5, 0.99, 50)

chi_emp,  chibar_emp  = calculate_tail_metrics(U_test[:, 0], U_test[:, 1], thresholds)
u_sim_g, v_sim_g      = gumbel.simulate(n=50000)
chi_mod,  chibar_mod  = calculate_tail_metrics(u_sim_g, v_sim_g, thresholds)
u_sim_t, v_sim_t      = t_cop.simulate(n=50000)
chi_t,    chibar_t    = calculate_tail_metrics(u_sim_t, v_sim_t, thresholds)

uv_diag      = np.column_stack([thresholds, thresholds]).astype(np.float32)
C_uu_neural  = copula_model.predict(uv_diag, verbose=0)[0].ravel()
chi_neural   = (1 - 2*thresholds + C_uu_neural) / (1 - thresholds)
p_both_n     = np.clip(1 - 2*thresholds + C_uu_neural, 1e-10, 1.0)
chibar_neural = (2 * np.log(1 - thresholds) / np.log(p_both_n)) - 1

# CHI PLOTS


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(thresholds, chi_emp,    'o', color='black',   alpha=0.4, ms=4, label='Empirical (Test)')
ax1.plot(thresholds, chi_mod,    '-', color='#e63946', lw=2, label=f'Gumbel (θ={gumbel.theta:.2f})')
ax1.plot(thresholds, chi_t,      '-', color='#1d3557', lw=2, label=f't-Copula (ν={t_cop.nu:.2f}, ρ={t_cop.rho:.2f})')
ax1.plot(thresholds, chi_neural, '--',color='#2a9d8f', lw=2, label='Neural Copula')
ax1.set_title(r'Tail Dependence $\chi(u)$', fontweight='bold')
ax1.set_xlabel('Threshold $u$'); ax1.set_ylabel(r'$\chi(u)$')
ax1.legend(); ax1.grid(alpha=0.2)

ax2.plot(thresholds, chibar_emp,    'o', color='black',   alpha=0.4, ms=4, label='Empirical (Test)')
ax2.plot(thresholds, chibar_mod,    '-', color='#e63946', lw=2, label='Gumbel')
ax2.plot(thresholds, chibar_t,      '-', color='#1d3557', lw=2, label='t-Copula')
ax2.plot(thresholds, chibar_neural, '--',color='#2a9d8f', lw=2, label='Neural Copula')
ax2.set_title(r'Residual Dependence $\bar{\chi}(u)$', fontweight='bold')
ax2.set_xlabel('Threshold $u$'); ax2.set_ylabel(r'$\bar{\chi}(u)$')
ax2.legend(); ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("chi_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → chi_comparison.png")

