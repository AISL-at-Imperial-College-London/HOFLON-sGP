"""Sparse GP Q-critic, Titsias/VFE collapsed bound, ARD Matern-5/2, inducing inputs FIXED at k-means centres.
Same split as learn_Q_gpytorch.py (runs 0-94 train, 95-99 held out), same features (no integrators),
same target (discounted return, gamma 0.9), constant prior mean = mean training return.
Hyperparameters: Adam (lr 0.02, n_iter iterations, final iterate kept; no closed-loop selection).
Output: critic file in the format of critics.py / sgp.py (kind "gp").
    python poly_learn_Q_sgp_matern.py --seed 0 --iters 500"""
import argparse, time, sys, json, numpy as np, pandas as pd, joblib
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sgp import SparseGP
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--iters", type=int, default=500)
ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--m_ind", type=int, default=500)
ap.add_argument("--data", default="data/multi_run_dataset.csv")
ap.add_argument("--out", default="models/q_sgp_mt52_sd{seed}.joblib")
a = ap.parse_args()
COLS = ["A", "I", "R", "P", "T", "err_P", "err_T", "u_I", "u_Tc"]; GAMMA, JIT = 0.9, 1e-6
df = pd.read_csv(a.data).sort_values(["run_id", "time_min"]).reset_index(drop=True)
G = np.zeros(len(df))
for _, g in df.groupby("run_id", sort=False):
    acc = 0.0
    for i in g.index[::-1]: acc = df.at[i, "reward"] + GAMMA * acc; G[i] = acc
X = df[COLS].values.astype(np.float64); tr = df.run_id.values < 95; te = ~tr
sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
m0, ys = float(G[tr].mean()), float(G[tr].std()); yn = (G - m0) / ys
Z = KMeans(a.m_ind, n_init=1, random_state=a.seed).fit(Xs[tr]).cluster_centers_
Xt, yt, Zj = jnp.asarray(Xs[tr]), jnp.asarray(yn[tr]), jnp.asarray(Z); N, M = Xt.shape[0], Z.shape[0]

def kern(A, B, ell, sf2):
    A_, B_ = A / ell, B / ell
    d2 = jnp.maximum((A_**2).sum(1)[:, None] + (B_**2).sum(1)[None, :] - 2 * A_ @ B_.T, 0.0)
    r = jnp.sqrt(5.0 * d2 + 1e-12)
    return sf2 * (1 + r + r**2 / 3.0) * jnp.exp(-r)

def neg_bound(p):
    ell, sf2, sn2 = jnp.exp(p["log_ell"]), jnp.exp(p["log_sf2"]), jnp.exp(p["log_sn2"])
    L = jnp.linalg.cholesky(kern(Zj, Zj, ell, sf2) + JIT * sf2 * jnp.eye(M))
    Kuf = kern(Zj, Xt, ell, sf2)                                   # M x N
    S = jax.scipy.linalg.solve_triangular(L, Kuf @ Kuf.T, lower=True)  # L^-1 Kuf Kfu
    AAT = jax.scipy.linalg.solve_triangular(L, S.T, lower=True) / sn2  # L^-1 Kuf Kfu L^-T / sn2 (symmetric)
    AAT = 0.5 * (AAT + AAT.T); LB = jnp.linalg.cholesky(jnp.eye(M) + AAT)
    Ay = jax.scipy.linalg.solve_triangular(L, Kuf @ yt, lower=True) / jnp.sqrt(sn2)
    c = jax.scipy.linalg.solve_triangular(LB, Ay, lower=True) / jnp.sqrt(sn2)
    b = (-0.5 * N * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(LB))) - 0.5 * N * jnp.log(sn2)
         - 0.5 * (yt @ yt) / sn2 + 0.5 * (c @ c) - 0.5 * (N * sf2 / sn2 - jnp.trace(AAT)))
    return -b / N

vg = jax.jit(jax.value_and_grad(neg_bound))
p = {"log_ell": jnp.full(9, np.log(2.0)), "log_sf2": jnp.array(0.0), "log_sn2": jnp.array(np.log(0.1))}
mo = jax.tree_util.tree_map(jnp.zeros_like, p); ve = jax.tree_util.tree_map(jnp.zeros_like, p)
b1, b2, eps = 0.9, 0.999, 1e-8; t0 = time.time(); hist = []
for it in range(1, a.iters + 1):
    f, g = vg(p)
    mo = jax.tree_util.tree_map(lambda m, gg: b1 * m + (1 - b1) * gg, mo, g)
    ve = jax.tree_util.tree_map(lambda v, gg: b2 * v + (1 - b2) * gg**2, ve, g)
    p = jax.tree_util.tree_map(lambda x, m, v: x - a.lr * (m / (1 - b1**it)) / (jnp.sqrt(v / (1 - b2**it)) + eps), p, mo, ve)
    hist.append(float(f))
    if it in (1, 2, 10) or it % 50 == 0:
        print(f"it {it:4d}  -bound/N {float(f):.5f}  sf2 {float(jnp.exp(p['log_sf2'])):.3f}  sn2 {float(jnp.exp(p['log_sn2'])):.4f}  [{time.time()-t0:.0f}s]", flush=True)
ell, sf2, sn2 = np.exp(np.asarray(p["log_ell"])), float(np.exp(p["log_sf2"])), float(np.exp(p["log_sn2"]))
gp = SparseGP(Z, ell, sf2, sn2, y_mean=m0, y_std=ys, kernel="matern52", jitter=JIT).fit(Xs[tr], G[tr])
mu, sd = gp.predict(Xs[te]); r2 = 1 - np.sum((G[te] - mu)**2) / np.sum((G[te] - G[te].mean())**2)
sdp = np.sqrt(sd**2 + sn2 * ys**2); cov = float(np.mean(np.abs(G[te] - mu) <= 1.96 * sdp))
info = dict(kernel="matern52", inducing="kmeans_fixed", m_ind=M, iters=a.iters, lr=a.lr, seed=a.seed, train_runs="0-94",
            r2_heldout=float(r2), cov95_heldout=cov, sd_med_heldout=float(np.median(sd)), sigma_f=float(np.sqrt(sf2) * ys),
            noise_sd=float(np.sqrt(sn2) * ys), ell=ell.tolist(), bound_hist=hist,
            sd_heldout_quantiles={q: float(np.quantile(sd, q)) for q in (0.5, 0.9, 0.99)})
joblib.dump(dict(kind="gp", gp=gp.state_dict(), scaler=sc, info=info, cols=COLS), a.out.format(seed=a.seed))
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items() if k not in ("bound_hist", "ell", "sd_heldout_quantiles")}))
print("ell", np.round(ell, 2))
