"""Per-run metrics for every results/<...>_Z_<config>_<member>/ folder."""
import glob, os, re, numpy as np, pandas as pd
from scipy.spatial import cKDTree
RES = "results"
DATA = "data/multi_run_dataset.csv"
F9 = ["A", "I", "R", "P", "T", "err_P", "err_T", "u_I", "u_Tc"]       # critic / AAE feature space
hist = pd.read_csv(DATA); tr = hist.run_id.values < 95
mu9, sd9 = hist.loc[tr, F9].mean().values, hist.loc[tr, F9].std(ddof=0).values
tree = cKDTree((hist.loc[tr, F9].values - mu9) / sd9)
d_ho = tree.query((hist.loc[~tr, F9].values - mu9) / sd9, k=10)[0].mean(1)
D99 = np.percentile(d_ho, 99)
I_RNG, TC_RNG = 2.5, 120.0          # kg/h, K per unit of scaled action

def run_metrics(g):
    g = g.sort_values("time_min"); t = g.time_min.values / 60.0
    P, T = g.P.values * 100, g["T"].values * 350
    du = np.abs(np.diff(g[["u_I", "u_Tc"]].values, axis=0)); late = t[1:] > 20
    d = tree.query((g[F9].values - mu9) / sd9, k=10)[0].mean(1)
    o = dict(reward=g.reward.sum(), Tmax=T.max(), Pmax=P.max(),
             rmse_eP=np.sqrt(np.mean(g.err_P**2)), rmse_eT=np.sqrt(np.mean(g.err_T**2)),
             TV_I=du[:, 0].sum() * I_RNG, TV_Tc=du[:, 1].sum() * TC_RNG,                 # kg/h, K over 100 h
             mv_I_late=du[late, 0].mean() * I_RNG, mv_Tc_late=du[late, 1].mean() * TC_RNG,  # mean |move| per step, t>20h
             P_late=P[t > 50].mean(), T_late=T[t > 50].mean(),
             hT360=(T > 360).sum() * 0.5, frac_ood=(d > D99).mean(), frac_ood_late=(d[t > 20] > D99).mean(), frac_ood_early=(d[t <= 20] > D99).mean(),
             Tc_min_late=TC_RNG * g.u_Tc.values[t > 20].min() + 280, Tc_max_late=TC_RNG * g.u_Tc.values[t > 20].max() + 280)
    o["Poff_late"] = abs(o["P_late"] - 100); o["Toff_late"] = abs(o["T_late"] - 350)
    if "Q_sd" in g: o["sd_late_med"] = np.median(g.Q_sd.values[t > 50])
    if "solve_ms" in g: o["solve_ms"] = g.solve_ms.mean()
    if "kappa" in g: o["kappa_late"] = g.kappa.values[t > 50].mean()
    return o

def collect():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "*_Z_*"))):
        m = re.search(r"_Z_([^_]+)_([^_]+)$", f)
        if not m: continue
        cfg, member = m.groups()
        fs = glob.glob(f + "/*.csv")
        if not fs: continue
        df = pd.concat(map(pd.read_csv, fs)).drop_duplicates(["run_id", "time_min"])
        for r, g in df.groupby("run_id"):
            if len(g) < 200: continue
            rows.append(dict(cfg=cfg, member=member, seed=11 + int(r), **run_metrics(g)))
    return pd.DataFrame(rows)

def historical():
    rows = []
    for r, g in hist.groupby("run_id"):
        rows.append(dict(cfg="historical", member=f"run{r}", seed=-1, **run_metrics(g)))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("held-out 9-D kNN distance d99 =", round(D99, 3))
    M = collect(); M.to_csv("tables/per_run.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(M.groupby("cfg")[["reward", "Tmax", "TV_I", "TV_Tc", "Poff_late", "frac_ood_late", "solve_ms"]].median().round(3))
