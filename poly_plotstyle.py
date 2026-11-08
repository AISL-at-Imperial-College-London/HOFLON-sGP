"""Plot style following the HOFLON repository scripts (poly_plot.py, poly_boxplot.py):
matplotlib defaults (DejaVu Sans), a title per panel, units on the y-axis, 'time [h]',
grid alpha 0.4, frameless legend across the top, named colours, boxplots with showmeans=True.
Font sizes are scaled for an IEEE two-column page."""
import glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1,
                     "savefig.dpi": 300, "pdf.fonttype": 42})
PAPER = "figures/"
RES = "results/"
DATA = "data/multi_run_dataset.csv"
# strategy -> (label, colour); repo colours: Historical gray, HOFLON blue, IQL orange
STRAT = {
    "historical":  ("Historical",               "gray"),
    "xgbnoAE":     ("XGBoost, no AAE",           "orange"),
    "xgbAAE":      ("HOFLON (XGBoost + AAE)",    "blue"),
    "xgbAAEw5000": ("XGBoost + AAE ×10$^4$",     "deepskyblue"),
    "sgpK0":       ("sGP, $\\kappa=0$",          "brown"),
    "sgpK0.5":     ("HOFLON-sGP, $\\kappa=0.5$", "mediumpurple"),
    "sgpK3":       ("sGP sq.-exp., $\\kappa=3$", "yellowgreen"),
    "mtK3":        ("HOFLON-sGP, $\\kappa=3$",   "green"),
}
SHORT = {"historical": "Historical", "xgbnoAE": "XGBoost\nno AAE", "xgbAAE": "HOFLON\n(XGB+AAE)",
         "xgbAAEw5000": "XGB+AAE\n×10$^4$", "sgpK0": "sGP\n$\\kappa$=0", "sgpK0.5": "sGP\n$\\kappa$=0.5", "sgpK3": "sGP\n$\\kappa$=3"}

def grid(ax): ax.grid(alpha=0.4, lw=0.5)

def load_runs(cfg, seeds=(11, 12, 13)):
    """all runs of a configuration as {(critic, seed): DataFrame} with physical columns"""
    out = {}
    if cfg == "historical":
        df = pd.read_csv(DATA)
        for r, g in df.groupby("run_id"): out[("hist", r)] = phys(g)
        return out
    for f in glob.glob(RES + f"*_Z_{cfg}_*"):
        if f.split("_Z_")[1].rsplit("_", 1)[0] != cfg: continue
        member = f.rsplit("_", 1)[1]
        fs = glob.glob(f + "/*.csv")
        if not fs: continue
        df = pd.concat(map(pd.read_csv, fs)).drop_duplicates(["run_id", "time_min"])
        for r, g in df.groupby("run_id"):
            if 11 + r in seeds and len(g) == 200: out[(member, 11 + r)] = phys(g.sort_values("time_min"))
    return out

def phys(g):
    g = g.copy()
    g["time_h"] = g.time_min / 60.0; g["Poly_kgm3"] = g.P * 100; g["Temp_K"] = g["T"] * 350
    g["Mono_kgm3"] = g.A * 100; g["f_I_kgh"] = 2.5 * g.u_I; g["T_c_K"] = 280 + 120 * g.u_Tc
    return g

def median_run(runs):
    """repo convention: sort runs by cumulative reward, take index len//2"""
    keys = sorted(runs, key=lambda k: runs[k].reward.sum())
    return keys[len(keys) // 2], keys[0]
