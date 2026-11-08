"""
poly_reactor_learn_Q_xgb_variants.py — the repo's XGBoost critic (poly_reactor_learn_Q.py
settings: 1600 trees, depth 8, lr 0.03, gamma 0.9, row-random 80/20 split), with
two options used in the robustness study:

  --random_state N   XGBoost's subsampling seed (the repo uses 42)
  --no_int           leave the integrator features out of the critic

    python poly_reactor_learn_Q_xgb_variants.py --random_state 0            # -> q_xgb_rs0.joblib
    python poly_reactor_learn_Q_xgb_variants.py --random_state 42 --no_int  # -> q_xgb_rs42_noint.joblib

Saved as {"kind": "xgb", "xgb": model, "cols": [...]}, which critics.py and the
controller (--mode xgb --xgb_file ...) read.
"""
import argparse
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

ap = argparse.ArgumentParser()
ap.add_argument("--random_state", type=int, default=42)
ap.add_argument("--no_int", action="store_true")
ap.add_argument("--csv", default="data/multi_run_dataset.csv")
args = ap.parse_args()

df = pd.read_csv(args.csv).sort_values(["run_id", "time_min"]).reset_index(drop=True)
G = np.zeros(len(df))
for _, grp in df.groupby("run_id", sort=False):
    g = 0.0
    for i in reversed(grp.index):
        g = df.at[i, "reward"] + 0.9 * g
        G[i] = g
cols = ["A", "I", "R", "P", "T", "err_P", "err_T", "int_err_P", "int_err_T", "u_I", "u_Tc"]
if args.no_int:
    cols = [c for c in cols if not c.startswith("int_")]
X, y = df[cols].values.astype(np.float32), G.astype(np.float32)
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.20, random_state=42)
m = XGBRegressor(objective="reg:squarederror", n_estimators=1600, learning_rate=0.03, max_depth=8,
                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist",
                 random_state=args.random_state).fit(Xtr, ytr)
name = f"models/q_xgb_rs{args.random_state}{'_noint' if args.no_int else ''}.joblib"
print(f"{name}: train R2 {r2_score(ytr, m.predict(Xtr)):.5f}, row-random val R2 {r2_score(yv, m.predict(Xv)):.5f}")
joblib.dump({"kind": "xgb", "xgb": m, "cols": cols}, name)
