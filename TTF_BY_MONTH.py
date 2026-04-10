




import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)
np.random.seed(0)
torch.manual_seed(0)

os.makedirs("./outputs", exist_ok=True)


# TTF MODEL  


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
    x_safe = x.clone(); x_safe[~inside] = 0.0
    bin_idx = torch.searchsorted(
        cum_w[..., 1:-1].contiguous(), x_safe.unsqueeze(-1)
    ).squeeze(-1).clamp(0, K - 1)
    def gather(t): return t.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    x_k  = gather(cum_w[..., :-1]); x_k1 = gather(cum_w[..., 1:])
    y_k  = gather(cum_h[..., :-1]); y_k1 = gather(cum_h[..., 1:])
    d_k  = gather(D[..., :-1]);     d_k1 = gather(D[..., 1:])
    s_k  = gather(H) / gather(W)
    xi    = ((x_safe - x_k) / (x_k1 - x_k + 1e-8)).clamp(0.0, 1.0)
    denom = s_k + (d_k + d_k1 - 2*s_k) * xi * (1 - xi)
    z_inside = y_k + (y_k1 - y_k) * (s_k*xi**2 + d_k*xi*(1-xi)) / (denom + 1e-8)
    log_jac_inside = (
        2*torch.log(s_k + 1e-8)
        + torch.log(d_k1*xi**2 + 2*s_k*xi*(1-xi) + d_k*(1-xi)**2 + 1e-8)
        - 2*torch.log(denom.abs() + 1e-8)
    )
    z       = torch.where(inside, z_inside, x)
    log_jac = torch.where(inside, log_jac_inside, torch.zeros_like(x))
    return z, log_jac


class RQSCouplingLayer(nn.Module):
    def __init__(self, num_bins=5, tail_bound=2.5):
        super().__init__()
        self.K = num_bins; self.tail_bound = tail_bound
        self.params = nn.Parameter(torch.randn(3*num_bins+1)*0.01)
    def forward(self, x):
        N = x.shape[0]; p = self.params.unsqueeze(0).expand(N, -1)
        z, lj = _rqs_forward(x.squeeze(-1), p[:,:self.K],
                              p[:,self.K:2*self.K], p[:,2*self.K:], self.tail_bound)
        return z.unsqueeze(-1), lj


class AffineLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.shift     = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        z  = torch.exp(self.log_scale)*x + self.shift
        lj = self.log_scale.expand(x.shape[0])
        return z, lj


class RQSFlow(nn.Module):
    def __init__(self, num_bins=5, tail_bound=2.5, depth=1):
        super().__init__()
        self.rqs_layers = nn.ModuleList([
            RQSCouplingLayer(num_bins, tail_bound) for _ in range(depth)])
        self.affine = AffineLayer()
    def forward(self, x):
        log_jac = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype); z = x
        for layer in self.rqs_layers:
            z, lj = layer(z); log_jac = log_jac + lj
        z, lj = self.affine(z); log_jac = log_jac + lj
        return z, log_jac


def ttf_inverse_torch(x, lam_pos, lam_neg, mu, sigma):
    y = (x - mu) / sigma
    s = torch.sign(y); s = torch.where(s == 0, torch.ones_like(s), s)
    lam_s = torch.where(s > 0, lam_pos*torch.ones_like(s), lam_neg*torch.ones_like(s))
    inner    = torch.clamp(lam_s*y.abs() + 1.0, min=1e-30)
    erfc_val = torch.clamp(inner**(-1.0/lam_s), min=1e-30, max=2.0-1e-7)
    return s * (-torch.erfinv(1.0 - erfc_val)) * (2.0**0.5)


def ttf_log_abs_jac_torch(z, lam_pos, lam_neg, sigma):
    s = torch.sign(z); s = torch.where(s == 0, torch.ones_like(s), s)
    lam_s = torch.where(s > 0, lam_pos*torch.ones_like(s), lam_neg*torch.ones_like(s))
    erfc_val = torch.clamp(torch.erfc(z.abs()/(2.0**0.5)), min=1e-30)
    return (torch.log(sigma)
            + (-lam_s-1.0)*torch.log(erfc_val)
            + torch.log(torch.tensor(2.0/(2.0*math.pi)**0.5))
            - 0.5*z**2)


# MONTH CONDITIONER


class MonthConditioner(nn.Module):
    def __init__(self, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(12, embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, month_idx):
        return self.mlp(self.embed(month_idx))          


# TTFRQS WITH MONTH CONDITIONING


class TTFRQS(nn.Module):
    name = "ttf"
    def __init__(self, num_bins=5, tail_bound=2.5, depth=1, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.flow        = RQSFlow(num_bins, tail_bound, depth)
        self.conditioner = MonthConditioner(embed_dim, hidden_dim)
        self.mu          = nn.Parameter(torch.tensor(0.0))
        self.log_sigma   = nn.Parameter(torch.tensor(0.0))
        lp_init = float(torch.distributions.Uniform(0.05, 1.0).sample([1]).item())
        ln_init = float(torch.distributions.Uniform(0.05, 1.0).sample([1]).item())
        self.log_lam_pos = nn.Parameter(torch.log(torch.tensor(lp_init)))
        self.log_lam_neg = nn.Parameter(torch.log(torch.tensor(ln_init)))

    def _ttf_params(self, month_idx):
        deltas = self.conditioner(month_idx)
        d_mu, d_ls, d_lp, d_ln = deltas.unbind(dim=-1)
        mu_m      = self.mu + d_mu
        sigma_m   = torch.exp(torch.clamp(self.log_sigma + d_ls, -6., 6.)).clamp(1e-3)
        lam_pos_m = torch.exp(torch.clamp(self.log_lam_pos + d_lp, -4., 4.)).clamp(0.02, 15.)
        lam_neg_m = torch.exp(torch.clamp(self.log_lam_neg + d_ln, -4., 4.)).clamp(0.02, 15.)
        return mu_m, sigma_m, lam_pos_m, lam_neg_m

    def log_prob(self, x, month_idx):
        mu_m, sigma_m, lam_pos_m, lam_neg_m = self._ttf_params(month_idx)
        z_ttf  = ttf_inverse_torch(x.squeeze(-1), lam_pos_m, lam_neg_m, mu_m, sigma_m)
        lj_ttf = -ttf_log_abs_jac_torch(z_ttf, lam_pos_m, lam_neg_m, sigma_m)
        z_base, lj_flow = self.flow(z_ttf.unsqueeze(-1))
        log_base = -0.5*(z_base.squeeze(-1)**2 + torch.log(torch.tensor(2*math.pi)))
        return log_base + lj_flow + lj_ttf

    def logpdf_np(self, x_np, month):
        self.eval()
        xt  = torch.tensor(x_np, dtype=torch.float32).view(-1, 1)
        mid = torch.full((len(x_np),), month - 1, dtype=torch.long)
        with torch.no_grad():
            lp = self.log_prob(xt, mid).numpy()
        return np.where(np.isfinite(lp), lp, -1e9)

    def ttf_params_np(self, month):
        """Return scalar TTF params for a given month (1-based), as numpy floats."""
        self.eval()
        mid = torch.tensor([month - 1], dtype=torch.long)
        with torch.no_grad():
            mu_m, sigma_m, lam_pos_m, lam_neg_m = self._ttf_params(mid)
        return (mu_m.item(), sigma_m.item(), lam_pos_m.item(), lam_neg_m.item())


# TRAINING LOOP


def _train_model(model, x_trn, m_trn, x_val, m_val,
                 n_epochs=2000, lr=5e-3, batch_size=512,
                 early_stop_patience=100, eval_period=20,
                 verbose=True, tag="model"):
    x_t = torch.tensor(x_trn, dtype=torch.float32).view(-1, 1)
    m_t = torch.tensor(m_trn, dtype=torch.long)
    x_v = torch.tensor(x_val, dtype=torch.float32).view(-1, 1)
    m_v = torch.tensor(m_val, dtype=torch.long)
    loader    = DataLoader(TensorDataset(x_t, m_t), batch_size=batch_size, shuffle=True)
    opt       = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    best_val  = -np.inf; patience = 0; best_state = None
    step = 0; data_iter = iter(loader)
    model.train()
    while step < n_epochs:
        try:    xb, mb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); xb, mb = next(data_iter)
        opt.zero_grad()
        lp   = model.log_prob(xb, mb)
        loss = -lp[torch.isfinite(lp)].mean()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step(); step += 1
        if step % eval_period == 0:
            model.eval()
            with torch.no_grad():
                val_lp = model.log_prob(x_v, m_v)
                val_ll = val_lp[torch.isfinite(val_lp)].mean().item()
            model.train()
            if val_ll > best_val:
                best_val = val_ll; patience = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if verbose and step % (eval_period*5) == 0:
                print(f"    [{tag}] step {step:5d}  val_ll={val_ll:.4f}")
            if patience >= early_stop_patience:
                if verbose: print(f"    [{tag}] early stop at step {step}")
                break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return model


class TTFMarginal:
    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    def __init__(self, model_cfg=None):
        self.model_cfg  = model_cfg or dict(num_bins=5, tail_bound=2.5, depth=1)
        self._model     = None
        self._cdf_grids = {}
        self._x_grid    = None

    @staticmethod
    def _to_month0(timestamps):
        ts = np.asarray(timestamps)
        if np.issubdtype(ts.dtype, np.integer):
            m1 = ts.ravel()
            return (m1 - 1).astype(np.int64)
        if np.issubdtype(ts.dtype, np.datetime64):
            return (pd.DatetimeIndex(ts).month.values - 1).astype(np.int64)
        ser = pd.to_datetime(pd.Series(timestamps))
        return (ser.dt.month.values - 1).astype(np.int64)

    def fit(self, x_np, timestamps,
            n_steps=12000, lr=5e-3, batch_size=512,
            early_stop_patience=100, eval_period=20,
            verbose=True, tag="ttf-marginal"):
        x_np   = np.asarray(x_np, dtype=np.float64)
        months = self._to_month0(timestamps)
        self._x_min = float(x_np.min()); self._x_max = float(x_np.max())
        self._model = TTFRQS(**self.model_cfg)
        idx = np.arange(len(x_np))
        trn_idx, val_idx = train_test_split(idx, test_size=0.15,
                                            random_state=0, stratify=months)
        _train_model(self._model,
                     x_np[trn_idx], months[trn_idx],
                     x_np[val_idx], months[val_idx],
                     n_epochs=n_steps, lr=lr, batch_size=batch_size,
                     early_stop_patience=early_stop_patience,
                     eval_period=eval_period, verbose=verbose, tag=tag)
        self._x_grid = np.linspace(self._x_min - 0.1, self._x_max + 0.1, 4000)
        for m in range(1, 13):
            pdf_grid = np.exp(self._model.logpdf_np(self._x_grid, m))
            cdf_grid = np.concatenate([[0.0],
                np.cumsum(np.diff(self._x_grid)*0.5*(pdf_grid[:-1]+pdf_grid[1:]))])
            cdf_grid = cdf_grid / (cdf_grid[-1] + 1e-12)
            self._cdf_grids[m] = np.clip(cdf_grid, 0., 1.)
        return self

    def predict(self, x_np, month, verbose=0):
        assert 1 <= month <= 12
        x_flat = np.asarray(x_np, dtype=np.float64).ravel()
        pdf = np.exp(self._model.logpdf_np(x_flat, month)).reshape(-1,1).astype(np.float32)
        neg = np.clip(-pdf, 0., None); pdf = np.clip(pdf, 1e-9, None)
        cdf = np.interp(x_flat, self._x_grid, self._cdf_grids[month]).reshape(-1,1).astype(np.float32)
        return [cdf, pdf, neg]

    def cdf(self, x_np, month): return self.predict(x_np, month)[0]
    def pdf(self, x_np, month): return self.predict(x_np, month)[1]

    def monthly_pdfs(self, x_np=None):
        if x_np is None: x_np = self._x_grid
        return {m: np.exp(self._model.logpdf_np(x_np, m)) for m in range(1, 13)}

    def monthly_params(self):
        """Return a DataFrame of per-month TTF parameters."""
        rows = []
        for m in range(1, 13):
            mu, sigma, lp, ln = self._model.ttf_params_np(m)
            rows.append(dict(month=m, month_name=self.MONTH_NAMES[m-1],
                             mu=mu, sigma=sigma, lam_pos=lp, lam_neg=ln))
        return pd.DataFrame(rows)




def plot_seasonal_pdfs(marginal, x_grid, obs, col_name, out_path):
    from scipy.stats import gaussian_kde

    pdfs   = marginal.monthly_pdfs(x_grid)
    colors = cm.twilight_shifted(np.linspace(0.05, 0.95, 12))
    fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True, sharey=False)
    fig.suptitle(f"Monthly TTF PDFs — {col_name}\n"
                 f"y-axis: probability density", fontsize=13, fontweight="bold")

    for ax, m in zip(axes.flat, range(1, 13)):
        name = TTFMarginal.MONTH_NAMES[m - 1]
        mask = (obs["month"] == m)
        vals = obs.loc[mask, "x"].values
        n    = len(vals)

        n_bins = max(8, int(np.sqrt(n)))
        counts, bin_edges, patches = ax.hist(
            vals, bins=n_bins, density=True,
            color="lightgrey", edgecolor="white", alpha=0.8)

        if n > 5:
            kde = gaussian_kde(vals, bw_method="scott")
            ax.plot(x_grid, kde(x_grid), lw=1.2, ls="--",
                    color="steelblue", alpha=0.7, label="KDE")

        ax.plot(x_grid, pdfs[m], lw=2, color=colors[m - 1], label="TTF")


        hist_top = counts.max() if len(counts) > 0 else 1.0
        ax.set_ylim(0, hist_top * 1.35)

        ax.set_title(f"{name}  (n={n})", fontsize=9)
        ax.set_xlabel(col_name, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))


    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")



# MAIN


if __name__ == "__main__":


    DATA_PATH = "wind_flood_loss_pairs_events2.0.csv"
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["month"] = df["date"].dt.month
    print(f"Loaded {len(df):,} rows. Month distribution:\n{df['month'].value_counts().sort_index()}\n")

    x_grid_wind  = np.linspace(df["log_wind"].min()  - 0.2, df["log_wind"].max()  + 0.2, 500)
    x_grid_flood = np.linspace(df["log_flood"].min() - 0.2, df["log_flood"].max() + 0.2, 500)


    print("=" * 60)
    print("Fitting TTFMarginal for log_wind …")
    print("=" * 60)
    marg_wind = TTFMarginal()
    marg_wind.fit(df["log_wind"].values, df["date"],
                  n_steps=12000, verbose=True, tag="log_wind")

    params_wind = marg_wind.monthly_params()
    print("\nPer-month TTF parameters (log_wind):")
    print(params_wind.to_string(index=False))

    plot_seasonal_pdfs(marg_wind, x_grid_wind,
                       df[["month", "log_wind"]].rename(columns={"log_wind": "x"}),
                       "log_wind", "./outputs/seasonal_pdfs_log_wind.png")


    print("\n" + "=" * 60)
    print("Fitting TTFMarginal for log_flood …")
    print("=" * 60)
    marg_flood = TTFMarginal()
    marg_flood.fit(df["log_flood"].values, df["date"],
                   n_steps=12000, verbose=True, tag="log_flood")

    params_flood = marg_flood.monthly_params()
    print("\nPer-month TTF parameters (log_flood):")
    print(params_flood.to_string(index=False))

    plot_seasonal_pdfs(marg_flood, x_grid_flood,
                       df[["month", "log_flood"]].rename(columns={"log_flood": "x"}),
                       "log_flood", "./outputs/seasonal_pdfs_log_flood.png")


    params_wind["variable"]  = "log_wind"
    params_flood["variable"] = "log_flood"
    params_all = pd.concat([params_wind, params_flood], ignore_index=True)
    params_all.to_csv("./outputs/monthly_params.csv", index=False)
    print("\nSaved: ./outputs/monthly_params.csv")
    print("\nDone.")

