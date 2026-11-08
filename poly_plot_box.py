"""Per-run distributions as in poly_boxplot.py (boxplot, showmeans=True): cumulative reward,
tracking RMSE of P and T, plus peak temperature. Seeds 11-13, five critics per controller (15 runs);
historical: the 100 logged PI start-ups."""
from poly_plotstyle import *
from matplotlib.patches import Patch
show = ["historical", "xgbnoAE", "xgbAAE", "xgbAAEw5000", "mtK3", "sgpK3"]
rows = []
for s in show:
    for k, g in load_runs(s).items():
        rows.append(dict(s=s, R=g.reward.sum(), eP=np.sqrt(np.mean(g.err_P**2)), eT=np.sqrt(np.mean(g.err_T**2)), Tmax=g.Temp_K.max()))
D = pd.DataFrame(rows)
panels = [("R", "Cumulative Reward", "Σ reward", (5000, 14000)),
          ("eP", "Tracking RMSE – P", "RMSE", (0.17, 0.27)),
          ("eT", "Tracking RMSE – T", "RMSE", (0.002, 0.0125)),
          ("Tmax", "Peak Reactor Temp.", "K", (352, 376))]
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.4))
for ax, (col, ttl, yl, lim) in zip(axs, panels):
    data = [D.loc[D.s == s, col].values for s in show]
    bp = ax.boxplot(data, showmeans=True, patch_artist=True, widths=0.6, whis=1.5,
                    medianprops=dict(color="k", lw=1.0),
                    meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="k", markersize=3.5, markeredgewidth=0.6),
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor="none", markeredgewidth=0.6),
                    boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for patch, s in zip(bp["boxes"], show):
        patch.set_facecolor(STRAT[s][1]); patch.set_alpha(0.55)
    ax.set_ylim(*lim); ax.set_title(ttl, pad=2); ax.set_ylabel(yl, labelpad=1); grid(ax)
    ax.set_xticks(range(1, len(show) + 1)); ax.set_xticklabels([str(i) for i in range(1, len(show) + 1)])
    # runs outside the plotted range are counted at the axis edge
    for i, d in enumerate(data, start=1):
        lo, hi = (d < lim[0]).sum(), (d > lim[1]).sum()
        if lo: ax.annotate(f"↓{lo}", (i, lim[0]), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6, color="red")
        if hi: ax.annotate(f"↑{hi}", (i, lim[1]), xytext=(0, -2), textcoords="offset points", ha="center", va="top", fontsize=6, color="red")
axs[3].axhline(360, ls="--", lw=0.7, c="k")
fig.legend(handles=[Patch(facecolor=STRAT[s][1], alpha=0.55, edgecolor="k", lw=0.5, label=f"{i}: {STRAT[s][0]}") for i, s in enumerate(show, 1)],
           loc="upper center", ncol=6, frameon=False, fontsize=6.2, handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
plt.tight_layout(rect=[0, 0, 1, 0.88], pad=0.3, w_pad=0.6)
plt.savefig(PAPER + "fig_box.pdf"); plt.savefig("figures/fig_box.png", dpi=200)
print(D.groupby("s")[["R", "eP", "eT", "Tmax"]].agg(["median", "min", "max"]).round(4).to_string())
