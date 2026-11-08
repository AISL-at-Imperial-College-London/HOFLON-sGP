"""Median-reward run per strategy, 2x3 layout of poly_plot.py."""
from poly_plotstyle import *
import matplotlib.gridspec as gridspec
show = ["historical", "xgbnoAE", "xgbAAE", "mtK3"]
fig = plt.figure(figsize=(7.16, 3.3))
gs = gridspec.GridSpec(2, 3, figure=fig)
ax = {k: fig.add_subplot(gs[i, j]) for k, (i, j) in
      {"poly": (0, 0), "temp": (0, 1), "rew": (0, 2), "fI": (1, 0), "Tc": (1, 1), "mono": (1, 2)}.items()}
info = {}
for s in show:
    runs = load_runs(s); (km, kw) = median_run(runs); sub = runs[km]; lab, c = STRAT[s]
    info[s] = (km, sub.reward.sum())
    st = dict(c=c, lw=1.0, alpha=0.85)
    ax["poly"].plot(sub.time_h, sub.Poly_kgm3, label=lab, **st); ax["temp"].plot(sub.time_h, sub.Temp_K, **st)
    ax["rew"].plot(sub.time_h, sub.reward, **st); ax["fI"].plot(sub.time_h, sub.f_I_kgh, **st)
    ax["Tc"].plot(sub.time_h, sub.T_c_K, **st); ax["mono"].plot(sub.time_h, sub.Mono_kgm3, **st)
ax["poly"].axhline(100, ls="--", lw=0.8, c="k"); ax["temp"].axhline(350, ls="--", lw=0.8, c="k")
lims = {"poly": (0, 150), "temp": (345, 362), "rew": (-150, 600), "fI": (0, 2.6), "Tc": (336, 350), "mono": (390, 560)}
titles = {"poly": "Polymer Conc.", "temp": "Reactor Temp.", "rew": "Reward", "fI": "Initiator Feed",
          "Tc": "Coolant Temp.", "mono": "Monomer A Conc."}
units = {"poly": "kg m$^{-3}$", "temp": "K", "rew": "Reward", "fI": "kg h$^{-1}$", "Tc": "K", "mono": "kg m$^{-3}$"}
for k, a in ax.items():
    a.set_ylim(*lims[k]); a.set_title(titles[k], pad=2); a.set_ylabel(units[k], labelpad=1); a.set_xlabel("time [h]", labelpad=1)
    a.set_xlim(0, 100); grid(a)
ax["Tc"].axhspan(280 + 120 * 0.501, 280 + 120 * 0.537, color="gray", alpha=0.15, lw=0)
fig.legend(loc="upper center", ncol=4, frameon=False, handlelength=1.6, columnspacing=1.2)
plt.tight_layout(rect=[0, 0, 1, 0.93], pad=0.3, h_pad=0.6, w_pad=0.8)
plt.savefig(PAPER + "fig_median.pdf"); plt.savefig("figures/fig_median.png", dpi=200)
for s, (k, R) in info.items(): print(s, k, round(R))
