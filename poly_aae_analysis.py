"""Size of the AAE term at the applied actions, and first divergence of paired runs with/without AAE."""
import glob, numpy as np, pandas as pd
RES = "results/"
def load(cfg, member):
    f = glob.glob(RES + f"*_Z_{cfg}_{member}")
    if not f: return None
    fs = glob.glob(f[0] + "/*.csv")
    return pd.concat(map(pd.read_csv, fs)).drop_duplicates(["run_id", "time_min"]) if fs else None
def pair_stats(cfgA, cfgB, members, lam1):
    rows = []
    for m in members:
        a, b = load(cfgA, m), load(cfgB, m)
        if a is None or b is None: continue
        for r in sorted(set(a.run_id) & set(b.run_id)):
            x, y = a[a.run_id == r].sort_values("time_min"), b[b.run_id == r].sort_values("time_min")
            if len(x) < 200 or len(y) < 200: continue
            d = np.abs(x[["u_I", "u_Tc"]].values - y[["u_I", "u_Tc"]].values).max(1)
            first = int(np.argmax(d > 1e-9)) if (d > 1e-9).any() else 200
            pen = lam1 * x.ae_recon.values
            rows.append(dict(member=m, seed=11 + r, first_div=first, pen_med=np.median(pen), pen_max=pen.max(),
                             RA=x.reward.sum(), RB=y.reward.sum()))
    return pd.DataFrame(rows)
if __name__ == "__main__":
    X = ["rs42", "rs0", "rs7", "rs1", "rs2"]; G = [f"f{i}" for i in range(5)]
    for A, B, mem, lam in [("xgbAAE", "xgbnoAE", X, 0.5 / 6), ("sgpK0AAEw0.5", "sgpK0", G, 0.5 / 6)]:
        P = pair_stats(A, B, mem, lam)
        if len(P):
            print(A, "vs", B, f"n={len(P)}  first divergence step: median {P.first_div.median():.0f}, min {P.first_div.min()}, max {P.first_div.max()}",
                  f"| AAE term at applied action: median {P.pen_med.median():.2e}, max {P.pen_max.max():.2e} return units")
            P.to_csv(f"pairs_{A}.csv", index=False)
