


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)



# RATIONAL QUADRATIC SPLINE


def _rqs_forward(x, widths, heights, derivatives, tail_bound):
    K = widths.shape[-1]
    W = torch.softmax(widths,  dim=-1) * 2 * tail_bound
    H = torch.softmax(heights, dim=-1) * 2 * tail_bound
    D = torch.nn.functional.softplus(derivatives) + 1e-5

    cum_w = torch.cat([torch.full_like(W[..., :1], -tail_bound),
                       -tail_bound + torch.cumsum(W, dim=-1)], dim=-1)
    cum_h = torch.cat([torch.full_like(H[..., :1], -tail_bound),
                       -tail_bound + torch.cumsum(H, dim=-1)], dim=-1)

    inside = (x >= -tail_bound) & (x <= tail_bound)
    x_safe = x.clone()
    x_safe[~inside] = 0.0

    bin_idx = torch.searchsorted(
        cum_w[..., 1:-1].contiguous(), x_safe.unsqueeze(-1)
    ).squeeze(-1).clamp(0, K - 1)

    def gather(t):
        return t.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    x_k, x_k1 = gather(cum_w[..., :-1]), gather(cum_w[..., 1:])
    y_k, y_k1 = gather(cum_h[..., :-1]), gather(cum_h[..., 1:])
    d_k, d_k1 = gather(D[..., :-1]),     gather(D[..., 1:])
    s_k       = gather(H) / gather(W)

    xi    = ((x_safe - x_k) / (x_k1 - x_k + 1e-8)).clamp(0.0, 1.0)
    denom = s_k + (d_k + d_k1 - 2 * s_k) * xi * (1 - xi)

    z_inside = y_k + (y_k1 - y_k) * (
        s_k * xi**2 + d_k * xi * (1 - xi)
    ) / (denom + 1e-8)

    log_jac_inside = (
        2 * torch.log(s_k + 1e-8)
        + torch.log(d_k1 * xi**2 + 2 * s_k * xi * (1-xi) + d_k * (1-xi)**2 + 1e-8)
        - 2 * torch.log(denom.abs() + 1e-8)
    )

    z       = torch.where(inside, z_inside, x)
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
        z, lj = _rqs_forward(x.squeeze(-1),
                              p[:, :self.K], p[:, self.K:2*self.K],
                              p[:, 2*self.K:], self.tail_bound)
        return z.unsqueeze(-1), lj


class AffineLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.shift     = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z  = torch.exp(self.log_scale) * x + self.shift
        lj = self.log_scale.expand(x.shape[0])
        return z, lj


class RQSFlow(nn.Module):
    def __init__(self, num_bins=5, tail_bound=2.5, depth=1):
        super().__init__()
        self.rqs_layers = nn.ModuleList([
            RQSCouplingLayer(num_bins, tail_bound) for _ in range(depth)
        ])
        self.affine = AffineLayer()

    def forward(self, x):
        log_jac = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = x
        for layer in self.rqs_layers:
            z, lj = layer(z); log_jac = log_jac + lj
        z, lj = self.affine(z); log_jac = log_jac + lj
        return z, log_jac

    def log_prob_gaussian(self, x):
        z, lj = self.forward(x)
        log_base = -0.5 * (z.squeeze(-1)**2 + torch.log(torch.tensor(2 * torch.pi)))
        return log_base + lj



# TTF TRANSFORMATION


def ttf_inverse_torch(x, lam_pos, lam_neg, mu, sigma):
    y     = (x - mu) / sigma
    s     = torch.sign(y)
    s     = torch.where(s == 0, torch.ones_like(s), s)
    lam_s = torch.where(s > 0, lam_pos * torch.ones_like(s),
                                lam_neg * torch.ones_like(s))
    inner    = torch.clamp(lam_s * y.abs() + 1.0, min=1e-30)
    erfc_val = torch.clamp(inner ** (-1.0 / lam_s), min=1e-30, max=2.0 - 1e-7)
    return s * (-torch.erfinv(1.0 - erfc_val)) * (2.0 ** 0.5)


def ttf_log_abs_jac_torch(z, lam_pos, lam_neg, sigma):
    s     = torch.sign(z)
    s     = torch.where(s == 0, torch.ones_like(s), s)
    lam_s = torch.where(s > 0, lam_pos * torch.ones_like(s),
                                lam_neg * torch.ones_like(s))
    erfc_val = torch.clamp(torch.erfc(z.abs() / (2.0**0.5)), min=1e-30)
    return (torch.log(sigma)
            + (-lam_s - 1.0) * torch.log(erfc_val)
           + torch.log(torch.tensor(2.0 / (2.0 * torch.pi)**0.5)))
           



class TTFRQS(nn.Module):
    name = "ttf"

    def __init__(self, num_bins=5, tail_bound=2.5, depth=1):
        super().__init__()
        self.flow      = RQSFlow(num_bins, tail_bound, depth)
        self.mu        = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        lp_init = float(torch.distributions.Uniform(0.05, 1.0).sample([1]).item())
        ln_init = float(torch.distributions.Uniform(0.05, 1.0).sample([1]).item())
        self.log_lam_pos = nn.Parameter(torch.log(torch.tensor(lp_init)))
        self.log_lam_neg = nn.Parameter(torch.log(torch.tensor(ln_init)))

    @property
    def lam_pos(self):
        return torch.exp(self.log_lam_pos).clamp(0.02, 15.0)

    @property
    def lam_neg(self):
        return torch.exp(self.log_lam_neg).clamp(0.02, 15.0)

    def log_prob(self, x):
        sigma         = torch.exp(self.log_sigma).clamp(1e-3)
        z_ttf         = ttf_inverse_torch(x.squeeze(-1), self.lam_pos,
                                          self.lam_neg, self.mu, sigma)
        lj_ttf        = -ttf_log_abs_jac_torch(z_ttf, self.lam_pos, self.lam_neg, sigma)
        z_base, lj_flow = self.flow(z_ttf.unsqueeze(-1))
        log_base      = -0.5 * (z_base.squeeze(-1)**2
                                + torch.log(torch.tensor(2 * torch.pi)))
        return log_base + lj_flow + lj_ttf

    def logpdf_np(self, x_np):
        self.eval()
        xt = torch.tensor(x_np, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            lp = self.log_prob(xt).numpy()
        return np.where(np.isfinite(lp), lp, -1e9)



def _train_model(model, x_trn, x_val,
                 n_epochs=2000, lr=5e-3, batch_size=512,
                 early_stop_patience=100, eval_period=20,
                 verbose=True, tag="model"):
    x_t    = torch.tensor(x_trn, dtype=torch.float32).view(-1, 1)
    x_v    = torch.tensor(x_val, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(x_t), batch_size=batch_size, shuffle=True)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    best_val, patience, best_state = -np.inf, 0, None
    step, data_iter = 0, iter(loader)

    model.train()
    while step < n_epochs:
        try:
            (xb,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (xb,) = next(data_iter)

        opt.zero_grad()
        lp   = model.log_prob(xb)
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
                val_lp = model.log_prob(x_v)
                val_ll = val_lp[torch.isfinite(val_lp)].mean().item()
            model.train()

            if val_ll > best_val:
                best_val   = val_ll
                patience   = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1

            if verbose and step % (eval_period * 5) == 0:
                print(f"    [{tag}] step {step:4d}  val_ll={val_ll:.4f}")

            if patience >= early_stop_patience:
                if verbose:
                    print(f"    [{tag}] early stop at step {step}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model



# Load wind_loss data
df  = pd.read_csv("wind_flood_loss_pairs2.0.csv")
raw = pd.to_numeric(df["wind_loss"], errors="coerce").dropna().values.astype(float)
rng = np.random.default_rng(42)
raw = rng.choice(raw[raw > 0], size=min(2000, len(raw[raw > 0])), replace=False)

losses_log = np.log(raw)
losses_log = losses_log[np.isfinite(losses_log)]

mu_all = losses_log.mean()
sd_all = losses_log.std()
x_all  = (losses_log - mu_all) / sd_all

print(f"Loaded {len(x_all):,} observations")


# Fit TTF 
MODEL_CFG = dict(num_bins=5, tail_bound=2.5, depth=1)
model = TTFRQS(**MODEL_CFG)
_train_model(model, x_all, x_all, tag="ttf-wind",
             n_epochs=10000, lr=5e-3, batch_size=512,
             early_stop_patience=100, eval_period=20, verbose=True)

print(f"\nFitted params: lam+={model.lam_pos.item():.4f}  "
      f"lam-={model.lam_neg.item():.4f}  "
      f"mu={model.mu.item():.4f}  "
      f"sigma={torch.exp(model.log_sigma).item():.4f}")




# Plot
xg       = np.linspace(x_all.min() - 0.3, x_all.max() + 0.3, 300)
lp       = model.logpdf_np(xg)
pdf_vals = np.exp(lp)
pdf_vals = pdf_vals / np.trapz(pdf_vals, xg)

p90 = np.quantile(x_all, 0.90)
xt  = np.linspace(p90, x_all.max() + 0.5, 200)
lp_tail = model.logpdf_np(xt)

counts, bin_edges = np.histogram(x_all, bins=60, density=True)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1
ax = axes[0]
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax.bar(bin_centres, counts, width=bin_edges[1]-bin_edges[0],
       color="#aec6cf", edgecolor="#5a7a85", linewidth=0.6, alpha=0.85)
ax.set_xlabel("log(Wind Loss) standardised", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("A · Data Histogram", fontsize=12, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

# Panel 2
ax2 = axes[1]
ax2.bar(bin_centres, counts, width=bin_edges[1]-bin_edges[0],
        color="#aec6cf", edgecolor="#5a7a85", linewidth=0.6, alpha=0.85, label="Data")
ax2.plot(xg, pdf_vals, lw=2.0, color="#9467bd", label="TTF")
ax2.set_xlabel("log(Wind Loss) standardised", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("B · Data + TTF Fit", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, frameon=False)
ax2.spines[["top","right"]].set_visible(False)

# Panel 3
ax3 = axes[2]
ax3.plot(xt, lp_tail, lw=2.0, color="#9467bd", label="TTF")
ax3.set_xlabel("log(Wind Loss) standardised", fontsize=11)
ax3.set_ylabel("Log-density", fontsize=11)
ax3.set_title("C · Upper Tail Log-Density (above 90th pct)", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9, frameon=False)
ax3.spines[["top","right"]].set_visible(False)

fig.suptitle(f"TTF on Wind Loss  (N = {len(x_all):,})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("ttf_wind_loss_plot.png", dpi=150, bbox_inches="tight")
plt.show()


