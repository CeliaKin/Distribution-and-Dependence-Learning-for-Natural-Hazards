#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import math
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os

os.makedirs('./initial_weights', exist_ok=True)
os.makedirs('./best_weights',    exist_ok=True)

# Reproducibility
from numpy.random import seed
seed(0)
tf.random.set_seed(0)

# Load data
df_raw = pd.read_csv("wind_flood_loss_pairs_events2.0.csv")
wind   = pd.to_numeric(df_raw["wind_loss"],  errors="coerce")
flood  = pd.to_numeric(df_raw["flood_loss"], errors="coerce")
df_clean = pd.DataFrame({"wind_loss": wind, "flood_loss": flood}).dropna()
print(f"Loaded {len(df_clean):,} rows")
print(df_clean.describe())

# Log-transform and normalise to [0,1] (matching original pipeline)
df_clean["log_wind"]  = np.log(df_clean["wind_loss"]  + 1)
df_clean["log_flood"] = np.log(df_clean["flood_loss"] + 1)

raw_data = df_clean[["log_wind", "log_flood"]].values.astype(np.float32)

# Normalise each dimension to [0,1]
data = raw_data.copy()
for i in range(data.shape[1]):
    col_min = data[:, i].min()
    col_max = data[:, i].max()
    data[:, i] = (data[:, i] - col_min) / (col_max - col_min + 1e-12)

X_train, X_test = train_test_split(data, test_size=0.33, random_state=42)

number_of_dimension        = 2
number_of_training_samples = X_train.shape[0]

data_domain = np.array([[X_train[:, i].min(), X_train[:, i].max()]
                         for i in range(number_of_dimension)], dtype=np.float32)

print(f"\nX_train shape: {X_train.shape}")
print(f"data_domain:\n{data_domain}")


# ===========================================================================
# CELL 2 — TTF MARGINAL CLASS
# ===========================================================================
# NOTE: TTFRQS and _train_model must already be defined from ttf_gpd_compare.py
# Paste or import them before running this cell.

class TTFMarginal:
    """
    Wraps TTFRQS to provide the same interface as the original Keras _marginal:
        .predict(x)  →  [cdf, pdf, neg]   (numpy arrays, shape (N,1))

    CDF is computed by numerical integration of the TTF PDF using the
    trapezoid rule on a fine grid, then looked up via linear interpolation.

    The model is fitted on the raw (normalised [0,1]) data — no additional
    standardisation is applied inside this class.
    """

    def __init__(self, model_cfg=None):
        self.model_cfg = model_cfg or dict(num_bins=5, tail_bound=2.5, depth=1)
        self._model    = None
        self._cdf_grid = None
        self._x_grid   = None

    def fit(self, x_np, n_steps=2000, lr=5e-3, batch_size=512,
            early_stop_patience=100, eval_period=20, verbose=True):
        """
        Fit TTF model.
        x_np : 1D numpy array of training values (already normalised)
        """
        x_np = np.asarray(x_np, dtype=np.float64)
        self._x_min = float(x_np.min())
        self._x_max = float(x_np.max())

        self._model = TTFRQS(**self.model_cfg)
        _train_model(self._model, x_np, x_np,
                     n_epochs=n_steps, lr=lr, batch_size=batch_size,
                     early_stop_patience=early_stop_patience,
                     eval_period=eval_period, verbose=verbose,
                     tag="ttf-marginal")

        # Precompute CDF by trapezoid integration on a fine grid
        self._x_grid = np.linspace(self._x_min - 0.1, self._x_max + 0.1, 4000)
        lp           = self._model.logpdf_np(self._x_grid)
        pdf_grid     = np.exp(lp)

        cdf_grid = np.concatenate([[0.0],
                       np.cumsum(np.diff(self._x_grid) *
                                 0.5 * (pdf_grid[:-1] + pdf_grid[1:]))])
        # normalise so CDF reaches exactly 1 at the right boundary
        cdf_grid       = cdf_grid / (cdf_grid[-1] + 1e-12)
        self._cdf_grid = np.clip(cdf_grid, 0.0, 1.0)
        return self

    def predict(self, x_np, verbose=0):
        """
        x_np : (N, 1) numpy array
        Returns [cdf, pdf, neg] each of shape (N, 1) as float32 — matching
        the interface expected by the Keras copula training code.
        """
        x_flat = np.asarray(x_np, dtype=np.float64).ravel()

        lp      = self._model.logpdf_np(x_flat)
        pdf     = np.exp(lp).reshape(-1, 1).astype(np.float32)
        neg     = np.clip(-pdf, 0.0, None)
        pdf_out = np.clip(pdf, 1e-9, None)

        cdf = np.interp(x_flat, self._x_grid, self._cdf_grid
                        ).reshape(-1, 1).astype(np.float32)
        return [cdf, pdf_out, neg]

    def cdf(self, x_np):
        return self.predict(x_np)[0]

    def pdf(self, x_np):
        return self.predict(x_np)[1]


# ===========================================================================
# CELL 3 — FIT TTF MARGINALS
# ===========================================================================

ttf_marginal_list = []
for i in range(number_of_dimension):
    print(f"\n── Fitting TTF marginal for dimension {i} ──")
    x_i = X_train[:, i].astype(np.float64)
    m   = TTFMarginal()
    m.fit(x_i, n_steps=12000, lr=5e-3, batch_size=512,
          early_stop_patience=100, eval_period=20, verbose=True)
    ttf_marginal_list.append(m)
    print(f"  Marginal {i} fitted.  "
          f"lam+={m._model.lam_pos.item():.3f}  "
          f"lam-={m._model.lam_neg.item():.3f}")

# Quick sanity-check plot
fig, axes = plt.subplots(1, number_of_dimension, figsize=(12, 4))
dim_names = ["wind_loss (normalised)", "flood_loss (normalised)"]
for i, m in enumerate(ttf_marginal_list):
    ax = axes[i]
    xg = np.linspace(data_domain[i, 0], data_domain[i, 1], 200).reshape(-1, 1)
    cdf_vals, pdf_vals, _ = m.predict(xg)
    ax.hist(X_train[:, i], bins=30, density=True,
            color="#aec6cf", edgecolor="#5a7a85", alpha=0.8, label="Data")
    ax.plot(xg, pdf_vals, "k-",  lw=1.5, label="TTF PDF")
    ax.plot(xg, cdf_vals, "k--", lw=1.5, label="TTF CDF")
    ax.set_title(f"TTF Marginal — dim {i} ({dim_names[i]})", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("ttf_marginals_check.png", dpi=150, bbox_inches="tight")
plt.show()
print("Marginal check plot saved.")


# ===========================================================================
# CELL 4 — COPULA ARCHITECTURE (unchanged from original)
# ===========================================================================

num_hidden_layers_for_copula  = 5
hidden_layer_width_for_copula = 10


class _copula(keras.Model):
    def __init__(self):
        super(_copula, self).__init__()
        self.kernel_regularizer = tf.keras.regularizers.L2(l2=0.001)
        self.get_components = Lambda(
            lambda x: [x[:, i:i+1] for i in range(number_of_dimension)])
        self.cat_components = Lambda(lambda x: K.concatenate(x, axis=-1))
        self.dense_layer_list = [
            keras.layers.Dense(hidden_layer_width_for_copula,
                               activation='tanh',
                               kernel_regularizer=self.kernel_regularizer)
            for _ in range(num_hidden_layers_for_copula)
        ]
        self.final_layer     = keras.layers.Dense(
            1, activation='sigmoid', kernel_regularizer=self.kernel_regularizer)
        self.gradient_layers = Lambda(
            lambda x: K.gradients(x[0], x[1]))
        self.relu = keras.layers.ReLU()

    def call(self, inputs):
        components = self.get_components(inputs)
        x = self.cat_components(components)
        for layer in self.dense_layer_list:
            x = layer(x)
        cdf = self.final_layer(x)
        pdf = cdf
        for i in range(number_of_dimension):
            pdf = self.gradient_layers([pdf, components[i]])
        neg = self.relu(-pdf[0])
        pdf = 1e-9 + self.relu(pdf[0])
        return [cdf, pdf, neg]


copula_model = _copula()

copula_prediction_input  = keras.layers.Input(shape=(number_of_dimension,))
copula_prediction_output = copula_model(copula_prediction_input)
copula_prediction_model  = keras.Model(inputs=copula_prediction_input,
                                        outputs=copula_prediction_output)


# ===========================================================================
# CELL 5 — PRECOMPUTE TTF CDFs FOR COPULA TRAINING
# ===========================================================================
# Because TTFMarginal is a PyTorch model outside the TF graph, we cannot
# backpropagate through it. Since marginals are fixed during copula training
# anyway, we precompute all required CDFs as numpy arrays once.

# CDFs of training data — used in the log-likelihood loss
joint_log_cdfs_np = np.column_stack([
    ttf_marginal_list[i].cdf(X_train[:, i].reshape(-1, 1)).ravel()
    for i in range(number_of_dimension)
]).astype(np.float32)

print(f"joint_log_cdfs_np shape: {joint_log_cdfs_np.shape}")
print(f"CDF ranges: "
      + "  ".join([f"dim{i}: [{joint_log_cdfs_np[:,i].min():.3f}, "
                   f"{joint_log_cdfs_np[:,i].max():.3f}]"
                   for i in range(number_of_dimension)]))


# ===========================================================================
# CELL 6 — COPULA TRAINING DATA
# ===========================================================================

number_boundary_points    = 400
number_partition_per_dim  = 50
number_observation_points = 100

# ── Boundary loss ──────────────────────────────────────────────────────────
joint_boundary_loss_data   = []
joint_boundary_loss_labels = []
for i in range(number_of_dimension):
    temp = np.random.rand(number_boundary_points,
                          number_of_dimension).astype(np.float32)
    temp[:, i] = 0
    joint_boundary_loss_data.append(temp)
    joint_boundary_loss_labels.append(temp[:, i])
for i in range(number_of_dimension):
    temp = np.ones([number_boundary_points, number_of_dimension],
                   dtype=np.float32)
    temp[:, i] = np.random.rand(number_boundary_points).astype(np.float32)
    joint_boundary_loss_data.append(temp)
    joint_boundary_loss_labels.append(temp[:, i])

joint_boundary_loss_data   = np.expand_dims(
    np.concatenate(joint_boundary_loss_data,   axis=0), axis=0)
joint_boundary_loss_labels = np.expand_dims(
    np.concatenate(joint_boundary_loss_labels, axis=0), axis=0)

# ── Neg / sum loss (uniform grid in CDF space) ────────────────────────────
grids = np.meshgrid(*[np.linspace(0, 1, number_partition_per_dim)
                       for _ in range(number_of_dimension)])
joint_neg_sum_loss_data = np.expand_dims(
    np.column_stack([g.ravel() for g in grids]).astype(np.float32), axis=0)
joint_neg_loss_labels = np.zeros([1, 1], dtype=np.float32)
joint_sum_loss_labels = np.ones( [1, 1], dtype=np.float32)

# ── Log-likelihood loss — uses precomputed TTF CDFs ───────────────────────
# Shape: (1, N, d) — the copula receives CDF values, not raw data
joint_log_cdfs_input_data = np.expand_dims(joint_log_cdfs_np, axis=0)
joint_log_loss_labels     = np.zeros([1, 1], dtype=np.float32) + 5.0

# ── Observation (empirical CDF) loss — also uses precomputed TTF CDFs ────
# Sample random points in [0,1]^d and compute empirical copula CDF
rng_obs = np.random.default_rng(42)
obs_u   = rng_obs.random(
    (number_observation_points, number_of_dimension)).astype(np.float32)
joint_observation_loss_data   = np.expand_dims(obs_u, axis=0)
joint_observation_loss_labels = np.zeros(
    [1, number_observation_points], dtype=np.float32)

# Empirical copula CDF: for each obs point u, count fraction of training
# CDFs that are <= u in all dimensions
for j, u in enumerate(obs_u):
    flags = np.all(joint_log_cdfs_np <= u, axis=1)
    joint_observation_loss_labels[0, j] = flags.mean()

print("Copula training data shapes:")
print(f"  boundary:    {joint_boundary_loss_data.shape}")
print(f"  neg/sum:     {joint_neg_sum_loss_data.shape}")
print(f"  log CDFs:    {joint_log_cdfs_input_data.shape}")
print(f"  observation: {joint_observation_loss_data.shape}")


# ===========================================================================
# CELL 7 — BUILD COPULA TRAINING MODEL
# ===========================================================================

joint_boundary_loss_input    = keras.layers.Input(
    shape=joint_boundary_loss_data.shape[1:])
joint_neg_sum_loss_input     = keras.layers.Input(
    shape=joint_neg_sum_loss_data.shape[1:])
joint_log_cdfs_input         = keras.layers.Input(
    shape=joint_log_cdfs_input_data.shape[1:])      # (N, d) — precomputed TTF CDFs
joint_observation_loss_input = keras.layers.Input(
    shape=joint_observation_loss_data.shape[1:])

joint_loss_input_list = [
    joint_boundary_loss_input,
    joint_neg_sum_loss_input,
    joint_log_cdfs_input,
    joint_observation_loss_input,
]

# Boundary loss — copula CDF at boundary points should equal marginal CDF
joint_boundary_loss_output = Lambda(
    lambda x: tf.expand_dims(x, axis=0), name="JB")(
    copula_model(joint_boundary_loss_input[0])[0][:, 0])

# Neg / sum losses — copula density should be non-negative and integrate to 1
_, joint_neg_sum_pdfs, joint_neg_sum_negs = copula_model(
    joint_neg_sum_loss_input[0])

joint_neg_loss_output = Lambda(
    lambda x: tf.math.reduce_sum(x, keepdims=True) /
              np.float32(joint_neg_sum_loss_input.shape[1]),
    name="JN")(joint_neg_sum_negs)

joint_sum_loss_output = Lambda(
    lambda x: tf.math.reduce_sum(x, keepdims=True) /
              np.float32(number_partition_per_dim - 1) ** number_of_dimension,
    name="JS")(joint_neg_sum_pdfs)

# Log-likelihood loss — copula density evaluated at precomputed TTF CDFs
# joint_log_cdfs_input[0] has shape (N, d) — direct input to copula
joint_log_loss_output = Lambda(
    lambda x: tf.math.reduce_sum(tf.math.log(x), keepdims=True) /
              np.float32(joint_log_cdfs_input_data.shape[1]),
    name="JL")(copula_model(joint_log_cdfs_input[0])[1])

# Observation loss — empirical vs copula CDF at random u points
joint_observation_loss_output = Lambda(
    lambda x: tf.expand_dims(x, axis=0), name="JO")(
    copula_model(joint_observation_loss_input[0])[0][:, 0])

joint_loss_output_list = [
    joint_boundary_loss_output,
    joint_neg_loss_output,
    joint_sum_loss_output,
    joint_log_loss_output,
    joint_observation_loss_output,
]

joint_model_train = keras.Model(
    inputs=joint_loss_input_list,
    outputs=joint_loss_output_list)

joint_training_input_data = [
    joint_boundary_loss_data,
    joint_neg_sum_loss_data,
    joint_log_cdfs_input_data,
    joint_observation_loss_data,
]

joint_training_labels = [
    joint_boundary_loss_labels,
    joint_neg_loss_labels,
    joint_sum_loss_labels,
    joint_log_loss_labels,
    joint_observation_loss_labels,
]

# Save initial weights
joint_model_train.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.0001), loss="mae")
joint_model_train.fit(
    x=joint_training_input_data, y=joint_training_labels,
    epochs=1, verbose=0)
joint_model_train.save_weights(
    './initial_weights/joint_model_initial_weights.h5')

temp_history    = joint_model_train.fit(
    x=joint_training_input_data, y=joint_training_labels,
    epochs=1, verbose=0)
joint_loss_keys = list(temp_history.history.keys())
print("Joint loss keys:", joint_loss_keys)


# ===========================================================================
# CELL 8 — COPULA TRAINING CALLBACK
# ===========================================================================

class Joint_Model_Training_Callback(tf.keras.callbacks.Callback):
    def __init__(self, record_interval=10, show_interval=500, verbose=1):
        super().__init__()
        self.loss_keys       = joint_loss_keys
        self.previous_loss   = 9999999
        self.record_interval = record_interval
        self.show_interval   = show_interval
        self.verbose         = verbose

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.record_interval == 0:
            current_losses = np.asarray([logs.get(k) for k in self.loss_keys])
            current_losses[-2] = 5.0 - current_losses[-2]   # log loss sign
            joint_epoch_number_list.append(epoch)
            joint_losses_list.append(current_losses)

            if current_losses[0] < self.previous_loss:
                self.previous_loss = current_losses[0]
                copula_model.save_weights(
                    './best_weights/copula_model_best_weights.h5')

            if self.verbose > 0 and epoch % (self.record_interval * 10) == 0:
                print(f"epoch: {epoch}")
                print([self.loss_keys[i] for i in range(len(self.loss_keys))])
                print([f"{current_losses[i]:.4f}"
                       for i in range(len(self.loss_keys))])

        if epoch % self.show_interval == 0:
            # Plot joint density using TTF marginals for the x/y axes
            domain_space = 50
            x_dom = np.linspace(0, 1, domain_space)
            y_dom = np.linspace(0, 1, domain_space)
            xm, ym = np.meshgrid(x_dom, y_dom)

            # Get marginal PDFs and CDFs from TTF models
            Fx = ttf_marginal_list[0].cdf(xm.ravel().reshape(-1, 1))
            fx = ttf_marginal_list[0].pdf(xm.ravel().reshape(-1, 1))
            Fy = ttf_marginal_list[1].cdf(ym.ravel().reshape(-1, 1))
            fy = ttf_marginal_list[1].pdf(ym.ravel().reshape(-1, 1))

            uv   = np.column_stack([Fx, Fy]).astype(np.float32)
            c_uv = copula_prediction_model.predict(uv, verbose=0)[1]
            f_xy = (c_uv * fx * fy).reshape(domain_space, domain_space)

            fig, ax = plt.subplots(figsize=(7, 6))
            cs = ax.contourf(xm, ym, f_xy, 100, cmap="viridis")
            fig.colorbar(cs, ax=ax, shrink=0.9)
            ax.set_xlabel("wind_loss (normalised)", fontsize=11)
            ax.set_ylabel("flood_loss (normalised)", fontsize=11)
            ax.set_title(f"Joint density f(x,y) — epoch {epoch}", fontsize=12)
            plt.tight_layout()
            plt.show()


# ===========================================================================
# CELL 9 — TRAIN COPULA
# ===========================================================================

joint_epoch_number_list = []
joint_losses_list       = []

joint_model_train.load_weights(
    './initial_weights/joint_model_initial_weights.h5')
joint_model_train.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.001),
    loss="mae",
    loss_weights=[2, 1, 1, 0.1, 5])

start_time = time.time()
joint_model_train.fit(
    x=joint_training_input_data,
    y=joint_training_labels,
    epochs=40000, verbose=0,
    callbacks=[Joint_Model_Training_Callback(
        record_interval=100, show_interval=5000, verbose=1)])
print(f"--- {time.time() - start_time:.1f} seconds ---")


# ===========================================================================
# CELL 10 — FINAL JOINT DENSITY PLOT
# ===========================================================================

domain_space = 100
x_dom = np.linspace(data_domain[0, 0], data_domain[0, 1], domain_space)
y_dom = np.linspace(data_domain[1, 0], data_domain[1, 1], domain_space)
xm, ym = np.meshgrid(x_dom, y_dom)

Fx = ttf_marginal_list[0].cdf(xm.ravel().reshape(-1, 1))
fx = ttf_marginal_list[0].pdf(xm.ravel().reshape(-1, 1))
Fy = ttf_marginal_list[1].cdf(ym.ravel().reshape(-1, 1))
fy = ttf_marginal_list[1].pdf(ym.ravel().reshape(-1, 1))

uv   = np.column_stack([Fx, Fy]).astype(np.float32)
c_uv = copula_prediction_model.predict(uv, verbose=0)[1]
f_xy = (c_uv * fx * fy).reshape(domain_space, domain_space)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Joint density contour
ax = axes[0]
cs = ax.contourf(xm, ym, f_xy, 100, cmap="viridis")
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.scatter(X_train[:, 0], X_train[:, 1],
           s=3, alpha=0.3, color="white", label="Training data")
ax.set_xlabel("wind_loss (normalised)", fontsize=11)
ax.set_ylabel("flood_loss (normalised)", fontsize=11)
ax.set_title("Joint density f(x,y) — TTF marginals + neural copula",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, frameon=False)

# Loss history
ax2 = axes[1]
losses_arr = np.array(joint_losses_list)
for j, key in enumerate(joint_loss_keys):
    ax2.plot(joint_epoch_number_list, losses_arr[:, j],
             lw=1.2, alpha=0.8, label=key)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Loss", fontsize=11)
ax2.set_title("Copula training losses", fontsize=11, fontweight="bold")
ax2.legend(fontsize=7, frameon=False)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("ttf_copula_joint_density.png", dpi=150, bbox_inches="tight")
plt.show()
print("Final plot saved -> ttf_copula_joint_density.png")


# In[ ]:


n_check  = 200
check_pts = np.random.rand(n_check, 2).astype(np.float32)

# Empirical joint CDF at raw data space points
emp_cdf = np.array([
    np.mean((X_test[:, 0] <= pt[0]) & (X_test[:, 1] <= pt[1]))
    for pt in check_pts
])

# Model joint CDF — pass through TTF marginals then copula
u_check = ttf_marginal_list[0].cdf(check_pts[:, 0:1])
v_check = ttf_marginal_list[1].cdf(check_pts[:, 1:2])
uv_check = np.column_stack([u_check, v_check]).astype(np.float32)
model_cdf = copula_prediction_model.predict(uv_check, verbose=0)[0].ravel()

rmse = np.sqrt(np.mean((emp_cdf - model_cdf)**2))
print(f"CDF RMSE: {rmse:.4f}  (lower is better, 0 = perfect)")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(emp_cdf, model_cdf, s=10, alpha=0.5, color="#9467bd")
ax.plot([0, 1], [0, 1], "r--", label="Perfect fit")
ax.set_xlabel("Empirical CDF", fontsize=11)
ax.set_ylabel("Model CDF",     fontsize=11)
ax.set_title(f"Empirical vs Fitted joint CDF\nRMSE = {rmse:.4f}", fontsize=11)
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.show()

