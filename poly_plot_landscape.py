"""Fig. 2: the four terms over the action set at one logged start-up state (historical median run, t = 6 h)."""
import sys, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")
from critics import load_critic
from np_torchload import load_pth
from poly_plotstyle import plt

PR = "models/"
hist = pd.read_csv("data/multi_run_dataset.csv")
R = hist.groupby("run_id").reward.sum(); run = int((R - R.median()).abs().idxmin()); t_h = 6.0
row = hist[(hist.run_id == run) & np.isclose(hist.time_min, t_h * 60)].iloc[0]
s7 = row[["A", "I", "R", "P", "T", "err_P", "err_T"]].values.astype(float)
ints = row[["int_err_P", "int_err_T"]].values.astype(float)
n1, n2 = 101, 126
uI, uT = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0.40, 0.65, n2)); U = np.column_stack([uI.ravel(), uT.ravel()])
X11 = np.hstack([np.tile(np.r_[s7, ints], (len(U), 1)), U])
Qx = load_critic(PR + "q_xgb_rs42.joblib").predict(X11)[0]
mu, sd = load_critic(PR + "q_sgp_mt52_sd0.joblib").predict(X11)
ck = load_pth(PR + "aae_weights.pth"); sc = joblib.load(PR + "ae_scaler.pkl")
E = [(ck["encoder"][f"net.{i}.weight"], ck["encoder"][f"net.{i}.bias"]) for i in (0, 2, 4)]
D = [(ck["decoder"][f"net.{i}.weight"], ck["decoder"][f"net.{i}.bias"]) for i in (0, 2, 4)]
def mlp(h, L):
    for j, (W, b) in enumerate(L):
        h = h @ W.T + b; h = np.tanh(h) if j < 2 else h
    return h
A9 = np.hstack([np.tile(s7, (len(U), 1)), U]); x = (A9 * sc.scale_ + sc.min_).astype(np.float32)
phi = np.mean((A9 - (mlp(mlp(x, E), D).astype(np.float64) - sc.min_) / sc.scale_) ** 2, 1)
lam1, kap = 0.5 / 6.0, 3.0
ua = hist.loc[np.isclose(hist.time_min, row.time_min), ["u_I", "u_Tc"]].values
panels = [("(a) XGBoost $\\hat Q$", Qx, "viridis", np.argmax(Qx - lam1 * phi)),
          ("(b) AAE $\\lambda_1\\phi_{\\rm lat}$", lam1 * phi, "magma_r", None),
          ("(c) sGP $\\mu$", mu, "viridis", np.argmax(mu - kap * sd)),
          ("(d) sGP $\\kappa\\sigma$, $\\kappa=3$", kap * sd, "magma_r", None)]
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.0), sharex=True, sharey=True)
for ax, (ttl, J, cm, kbest) in zip(axs.flat, panels):
    lo, hi = np.percentile(J, [1, 99.5]) if "XGB" in ttl or "mean" in ttl else (J.min(), J.max())
    im = ax.imshow(J.reshape(n2, n1), origin="lower", extent=[0, 1, 0.40, 0.65], aspect="auto", cmap=cm, vmin=lo, vmax=hi)
    ax.scatter(ua[:, 0], ua[:, 1], s=2.5, c="w", edgecolors="k", linewidths=0.2)
    if kbest is not None:
        ax.plot(U[kbest, 0], U[kbest, 1], marker="*", ms=7, mfc="r", mec="k", mew=0.4)
    if "mean" in ttl:
        k0 = np.argmax(mu); ax.plot(U[k0, 0], U[k0, 1], marker="o", ms=4, mfc="none", mec="r", mew=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02); cb.ax.tick_params(labelsize=5.5)
    from matplotlib.ticker import ScalarFormatter, MaxNLocator
    f = ScalarFormatter(useOffset=False); f.set_scientific(False); cb.formatter = f
    cb.locator = MaxNLocator(4); cb.update_ticks()
    ax.set_title(ttl, pad=2)
for ax in axs[1]: ax.set_xlabel("$u_I$", labelpad=1)
for ax in axs[:, 0]: ax.set_ylabel("$u_{T_c}$", labelpad=1)
plt.tight_layout(pad=0.25, w_pad=1.2, h_pad=0.8)
plt.savefig("figures/fig_landscape.pdf"); plt.savefig("figures/fig_landscape.png", dpi=220)
info = dict(run=run, t=t_h, P=row.P * 100, T=row["T"] * 350, Qrange=float(Qx.max() - Qx.min()), Qmin=Qx.min(), Qmax=Qx.max(),
            aae_min=float((lam1 * phi).min()), aae_max=float((lam1 * phi).max()), musd=(mu.min(), mu.max()), sdmin=sd.min(), sdmax=sd.max(),
            argmax_mu=U[np.argmax(mu)], argmax_lcb=U[np.argmax(mu - kap * sd)], argmax_xgb=U[np.argmax(Qx - lam1 * phi)],
            logged=(row.u_I, row.u_Tc))
for k, v in info.items(): print(k, np.round(v, 4) if not isinstance(v, int) else v)

import json
fmt = lambda v: f"{v:,.0f}".replace(",", "{,}") if v >= 10 else f"{v:.2g}"
ks = kap * sd
json.dump(dict(LandAAEmax=f"{float((lam1*phi).max()-(lam1*phi).min()):.2g}", LandQrange=fmt(float(Qx.max()-Qx.min())),
               LandSdRange=fmt(float(ks.min())) + "--" + fmt(float(ks.max())), LandMuRange=fmt(float(mu.max()-mu.min()))),
          open("tables/landscape.json", "w"))
print(open("tables/landscape.json").read())
