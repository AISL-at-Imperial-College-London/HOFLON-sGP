"""Effect of kappa: Matern-5/2 critics (HOFLON-sGP) and squared-exponential critics, five critics each,
seeds 11-13; HOFLON (XGBoost + AAE) on the same runs as reference."""
from poly_plotstyle import *
M = pd.read_csv("tables/per_run.csv"); M = M[M.seed <= 13]
def sweep(prefix, kmin=0.0):
    K = M[M.cfg.str.fullmatch(prefix + r"[0-9.]+")].copy(); K["kappa"] = K.cfg.str[len(prefix):].astype(float)
    return K[K.kappa >= kmin].groupby("kappa")
gS, gM = sweep("sgpK"), sweep("mtK", 0.5)
ref = M[M.cfg == "xgbAAE"]
cM, cS, cB = STRAT["mtK3"][1], STRAT["sgpK3"][1], STRAT["xgbAAE"][1]
P = [("reward", 1e-3, "Cumulative Reward", "Σ reward [×10$^3$]", False),
     ("mv_Tc_late", 1, "Coolant Move per Step", "K", True),
     ("Poff_late", 1, "Polymer Offset", "kg m$^{-3}$", True),
     ("frac_ood_late", 100, "Off-Support Steps", "%", False)]
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.15), sharex=True)
for ax, (col, sc, ttl, yl, logy) in zip(axs.flat, P):
    r = ref[col].quantile([0, .5, 1]) * sc
    ax.axhline(r[0.5], color=cB, lw=0.9, ls="--", label="HOFLON (XGBoost + AAE)")
    for g, c, lab, mk in [(gS, cS, "sGP, squared-exponential", "o"), (gM, cM, "HOFLON-sGP, Matérn-5/2", "s")]:
        q = g[col].quantile([0, .5, 1]).unstack() * sc
        lo = np.maximum(q[0], 1e-4) if logy else q[0]
        ax.fill_between(q.index, lo, q[1], color=c, alpha=0.22, lw=0)
        ax.plot(q.index, np.maximum(q[0.5], 1e-4) if logy else q[0.5], marker=mk, ls="-", color=c, ms=2.6, lw=1.0, label=lab)
        if 3.0 in q.index: ax.plot([3], [max(q.loc[3.0, 0.5], 1e-4) if logy else q.loc[3.0, 0.5]], marker="*", ms=7, color=c, mec="k", mew=0.4, zorder=5)
    if logy: ax.set_yscale("log")
    ax.set_title(ttl, pad=2); ax.set_ylabel(yl, labelpad=1); grid(ax); ax.set_xlim(-0.25, 5.25)
axs[0, 0].axhline(6, color="k", lw=0.6, ls=":"); axs[0, 0].set_ylim(2, 14.5)
for ax in axs[1]: ax.set_xlabel("confidence level κ", labelpad=1)
h, l = axs[0, 0].get_legend_handles_labels()
order = [l.index(x) for x in ["HOFLON-sGP, Matérn-5/2", "sGP, squared-exponential", "HOFLON (XGBoost + AAE)"]]
fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=2, frameon=False, fontsize=6.0,
           handlelength=1.8, columnspacing=0.8)
plt.tight_layout(rect=[0, 0, 1, 0.885], pad=0.3, h_pad=0.5, w_pad=0.6)
plt.savefig(PAPER + "fig_kappa.pdf"); plt.savefig("figures/fig_kappa.png", dpi=220)
