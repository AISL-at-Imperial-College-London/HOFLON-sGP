"""
poly_reactor_hoflon_cntrl_SGP.py — HOFLON online layer for the polymerisation
CSTR with a selectable critic.  Plant, noise, reward, objective weights
(f1=0.5 AAE, f2=6 Q, f3=0.025 move), solver portfolio (Nelder-Mead 250,
bounded Powell 250, DIRECT 40) and anti-windup are as in
poly_reactor_hoflon_cntrl.py.  Only the critic term changes:

    J(a) = f1 * recon_mse(s,a) - f2 * [mu(s,a) - penalty(sd(s,a))] + f3 * ||a - a_prev||^2

--mode
    xgb        repo XGBoost critic (q_discounted.joblib), sd = 0
    mean       sparse GP mean, kappa = 0
    lcb        kappa = --kappa_lcb (1.96)
    hybrid_s   kappa ramps with the distance of the observed STATE from the
               training-state support (fixed within a step)
    hybrid_sa  kappa ramps with the distance of the candidate (s, a) from the
               training state-action support (varies with a)
    trust      no penalty while sd <= sd_thr, then kappa_trust per unit of sd above it:
               penalty = --kappa_trust * max(0, sd - sd_thr); sd_thr defaults to the
               99th percentile of the critic's sd on held-out runs (stored by
               learn_Q_resgp.py), or set --sd_thr
    penalty = kappa * sd in the other GP modes, with
    kappa(d) = KMIN + (KMAX-KMIN) * clip((d - d50)/(d99 - d50), 0, 1), d = mean
    distance to the 10 nearest training points (standardised features); d50/d99
    from held-out runs 95-99.

--gp_file   any critic file with an sd: q_resgp.joblib (XGBoost + residual GP,
            learn_Q_resgp.py) or q_gpt_sgpr_*.joblib (learn_Q_gpytorch.py)
--runs N    runs differ only through measurement noise (fixed initial state, as
            in the repo); run r uses seed 1000+r in every configuration.

Output: results/<name>/runs_<first>_<last>.csv with the repo's column names
(run_id, time_min, A, I, R, P, T, err_P, err_T, int_err_P, int_err_T, u_I, u_Tc,
reward) plus Q_mu, Q_sd, kappa, d_state, d_sa, ae_recon.  int_err_* are logged
divided by 1000 as in multi_run_dataset.csv.

    python poly_reactor_hoflon_cntrl_SGP.py --mode xgb --runs 20
    python poly_reactor_hoflon_cntrl_SGP.py --mode lcb --gp_file q_gpt_sgpr_pessimistic.joblib --runs 20
    python poly_reactor_hoflon_cntrl_SGP.py --mode trust --gp_file q_resgp.joblib --ae none --runs 20
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, pandas as pd, joblib
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgp import SparseGP  # noqa: E402
try:
    from scipy.optimize import direct
    HAVE_DIRECT = True
except ImportError:
    HAVE_DIRECT = False

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="mean", choices=["xgb", "mean", "lcb", "hybrid_s", "hybrid_sa", "trust"])
ap.add_argument("--gp_file", default="q_gpt_sgpr_pessimistic.joblib")
ap.add_argument("--runs", type=int, default=20)
ap.add_argument("--first_run", type=int, default=0)
ap.add_argument("--hours", type=float, default=100.0)
ap.add_argument("--kappa_lcb", type=float, default=1.96)
ap.add_argument("--no_gp_mean", action="store_true",
                help="residual critic only: mean = XGBoost, the GP supplies only sd")
ap.add_argument("--kappa_trust", type=float, default=3.0, help="trust mode: penalty slope above sd_thr")
ap.add_argument("--sd_thr", type=float, default=None,
                help="trust mode: sd threshold in return units (default: held-out q99 stored in the critic file)")
ap.add_argument("--kappa_min", type=float, default=0.0)
ap.add_argument("--kappa_max", type=float, default=3.0)
ap.add_argument("--w_ae", type=float, default=0.5, help="f1 (repo: 0.5)")
ap.add_argument("--w_q", type=float, default=6.0, help="f2 (repo: 6)")
ap.add_argument("--w_du", type=float, default=0.025, help="f3 (repo: 0.025)")
ap.add_argument("--no_direct", action="store_true", help="drop DIRECT from the portfolio")
ap.add_argument("--ae", default="aae", choices=["aae", "none", "standin"],
                help="aae: repo AAE; none: no manifold penalty (ablation)")
ap.add_argument("--data_csv", default="data/multi_run_dataset.csv")
ap.add_argument("--xgb_file", default="q_discounted.joblib", help="XGBoost critic for --mode xgb")
ap.add_argument("--seed_base", type=int, default=1000,
                help="run r uses np.random.default_rng(seed_base + r)")
ap.add_argument("--tag", default="")
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--opt", default="portfolio", choices=["portfolio", "grid"],
                help="portfolio: repo NM + Powell + DIRECT; grid: score an n x n grid over the action box "
                     "(vectorised critic), then refine the best point with Nelder-Mead (only with --ae none)")
ap.add_argument("--n_grid", type=int, default=31)
ap.add_argument("--clip_int", action="store_true",
                help="clip the integrator features to their training range before the critic sees them")
ap.add_argument("--u_box", default="full", choices=["full", "data"],
                help="action search box: full [0,1]^2 (repo) or the range of actions in the training data")
ap.add_argument("--kappa_ss", type=float, default=None,
                help="corridor-triggered kappa (HOFLON Alg. 1 weight tightening): kappa_lcb until |e_P|,|e_T| < "
                     "--corr_tol for --corr_n consecutive steps, then kappa_ss; back to kappa_lcb on leaving")
ap.add_argument("--corr_tol", type=float, default=0.04, help="corridor half-width on scaled errors (repo pick_weights)")
ap.add_argument("--corr_n", type=int, default=4, help="consecutive in-corridor steps before tightening (repo)")
args = ap.parse_args()

def _gp_name(f):
    b = os.path.basename(f).replace(".joblib", "")
    return b[6:] if b.startswith("q_sgp_") else (b[2:] if b.startswith("q_") else b)
gp_tag = "" if args.mode == "xgb" else "_" + _gp_name(args.gp_file)
NAME = f"{args.mode}{gp_tag}{'_grid' if args.opt == 'grid' else ''}{'_ci' if args.clip_int else ''}{'_box' if args.u_box == 'data' else ''}{'_sdonly' if args.no_gp_mean else ''}{'' if args.ae != 'none' else '_noAE'}{f'_kss{args.kappa_ss:g}' if args.kappa_ss is not None else ''}{args.tag}"
OUT_DIR = os.path.join("results", NAME); os.makedirs(OUT_DIR, exist_ok=True)
LAST = args.first_run + args.runs - 1
CSV_OUT = os.path.join(OUT_DIR, f"runs_{args.first_run:03d}_{LAST:03d}.csv")

# ───────────── 1. plant (copied from poly_reactor_hoflon_cntrl.py) ─────────
R_GAS = 8.314; RHO = 1_000.0; CP = 2_000.0
UA, V = 550.0, 1.0
W_A, W_S, T_FEED = 100.0, 80.0, 350.0
MW_A, MW_I = 0.100, 0.165
A_d, E_d, A_p, E_p, A_t, E_t = 1e9, 125e3, 4e4, 25e3, 1e6, 15e3
DELTA_H = 100e3
TC_MIN, TC_MAX = 280.0, 400.0
I_MIN, I_MAX = 0.0, 2.5
S_CONC, S_TEMP = 100.0, 350.0

def _k(A, E, T): return A * np.exp(-E / (R_GAS * T))

def cstr_ode_scaled(t, y5, u_I, Tc_s):
    A, I, R, P, T = (*y5[:4] * S_CONC, y5[4] * S_TEMP)
    f_I = I_MIN + u_I * (I_MAX - I_MIN)
    T_c = TC_MIN + Tc_s * (TC_MAX - TC_MIN)
    F = (f_I + W_A + W_S) / (RHO * 3600.0); D = F / V
    k_d, k_p, k_t = _k(A_d, E_d, T), _k(A_p, E_p, T), _k(A_t, E_t, T)
    r_d = k_d * I / MW_I; r_p = k_p * R * A / MW_A; r_t = k_t * R**2
    dA = (W_A / 3600) / V - D * A - r_p * MW_A
    dI = (f_I / 3600) / V - D * I - r_d * MW_I
    dR = 2 * r_d - 2 * r_t
    dP = r_p * MW_A - D * P
    dT = (DELTA_H * r_p * V - UA * (T - T_c) + RHO * CP * F * (T_FEED - T)) / (RHO * CP * V)
    return [dA / S_CONC, dI / S_CONC, dR / S_CONC, dP / S_CONC, dT / S_TEMP]

def compute_reward(poly_t, T_s_t, poly_prev, T_s_prev, setpts, sigma=0.05):
    err_now = (poly_t - setpts["P"])**2 + (T_s_t - setpts["T_s"])**2
    err_prev = (poly_prev - setpts["P"])**2 + (T_s_prev - setpts["T_s"])**2
    return 50 * (-err_now + np.exp(-err_now / sigma**2)) + 5_000 * (err_prev - err_now)

SIG_MONO_S, SIG_POLY_S, SIG_TEMP_S = 2.0 / S_CONC, 0.5 / S_CONC, 0.2 / S_TEMP
SETPT = {"P": 1.0, "T_s": 1.0}
Y0 = np.array([5.55555456, 0.0, 0.0, 0.0, 1.0])
DT = 30 * 60

# ───────────── 2. critic ───────────────────────────────────────────────────
COLS = ["A", "I", "R", "P", "T", "err_P", "err_T", "int_err_P", "int_err_T", "u_I", "u_Tc"]
hist = pd.read_csv(args.data_csv)
tr = hist["run_id"].values < 95
INT_LO = hist.loc[tr, ["int_err_P", "int_err_T"]].min().values
INT_HI = hist.loc[tr, ["int_err_P", "int_err_T"]].max().values
if args.u_box == "data":
    U_LO, U_HI = hist.loc[tr, ["u_I", "u_Tc"]].min().values, hist.loc[tr, ["u_I", "u_Tc"]].max().values
else:
    U_LO, U_HI = np.zeros(2), np.ones(2)
print(f"[setup] clip_int={args.clip_int} (int range {INT_LO.round(3)}..{INT_HI.round(3)}); "
      f"action box u_I {U_LO[0]:.3f}-{U_HI[0]:.3f}, u_Tc {U_LO[1]:.3f}-{U_HI[1]:.3f}")
if args.mode == "xgb":
    Q_model = joblib.load(args.xgb_file)                  # raw features, as in the repo
    if isinstance(Q_model, dict):                         # XGBoost on a column subset (critics.py format)
        from critics import load_critic
        _xc = load_critic(args.xgb_file)
        class _Wrap:
            def predict(self, X): return _xc.predict(np.atleast_2d(X))[0]
        Q_model = _Wrap()
    dist_scaler = StandardScaler().fit(hist.loc[tr, COLS].values)
    def critic(sa_raw):
        return float(Q_model.predict(sa_raw.reshape(1, -1))[0]), 0.0
else:
    from critics import load_critic
    CRIT = load_critic(args.gp_file)
    if not CRIT.has_sd:
        raise SystemExit(f"{args.gp_file} has no uncertainty; use --mode xgb for a plain XGBoost critic")
    dist_scaler = CRIT.scaler if len(CRIT.idx) == len(COLS) else StandardScaler().fit(hist.loc[tr, COLS].values)
    def critic(sa_raw):
        mu, sd = CRIT.predict(sa_raw.reshape(1, -1), gp_mean=not args.no_gp_mean)
        return float(mu[0]), float(sd[0])
    if args.mode == "trust":
        SD_THR = args.sd_thr if args.sd_thr is not None else CRIT.sd_quantiles.get("q99")
        if SD_THR is None:
            raise SystemExit("trust mode: no held-out sd quantiles in the critic file; pass --sd_thr")
        print(f"[trust] sd threshold {SD_THR:.2f}, slope {args.kappa_trust}")

# ───────────── 3. support distances (logged in every mode) ─────────────────
H = dist_scaler.transform(hist[COLS].values)
K_NN = 10
tree_s, tree_sa = cKDTree(H[tr, :9]), cKDTree(H[tr])
d_ho_s = tree_s.query(H[~tr, :9], k=K_NN)[0].mean(1)
d_ho_sa = tree_sa.query(H[~tr], k=K_NN)[0].mean(1)
S50, S99 = np.percentile(d_ho_s, [50, 99]); SA50, SA99 = np.percentile(d_ho_sa, [50, 99])
mu_x, sd_x = dist_scaler.mean_, dist_scaler.scale_
def knn(tree, z): return float(np.mean(tree.query(z, k=K_NN)[0]))
def ramp(d, d50, d99):
    return args.kappa_min + (args.kappa_max - args.kappa_min) * float(np.clip((d - d50) / (d99 - d50), 0, 1))
print(f"[{NAME}] support: state d50={S50:.3f} d99={S99:.3f} | state-action d50={SA50:.3f} d99={SA99:.3f}")

# ───────────── 4. AAE (9-D: A,I,R,P,T,err_P,err_T,u_I,u_Tc) ───────────────
if args.ae == "aae":
    try:
        import torch, torch.nn as nn
        torch.set_num_threads(1)
        enc = nn.Sequential(nn.Linear(9, 64), nn.Tanh(), nn.Linear(64, 16), nn.Tanh(), nn.Linear(16, 4))
        dec = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 9))
        ck = torch.load("aae_weights.pth", map_location="cpu")
        enc.load_state_dict({k.replace("net.", ""): v for k, v in ck["encoder"].items()})
        dec.load_state_dict({k.replace("net.", ""): v for k, v in ck["decoder"].items()})
        enc.eval(); dec.eval()
        AE_SCALER = joblib.load("models/ae_scaler.pkl")
        def recon_mse(ae_in):                     # error in ORIGINAL units, as in the repo
            x = AE_SCALER.transform(ae_in.reshape(1, -1))
            with torch.no_grad():
                r = dec(enc(torch.from_numpy(x).float())).numpy()
            return float(np.mean((ae_in - AE_SCALER.inverse_transform(r)[0])**2))
        print("[setup] AAE: torch")
    except ImportError:                           # torch-free forward pass (float32, same weights)
        from np_torchload import load_pth
        ck = load_pth("models/aae_weights.pth")
        E = [(ck["encoder"][f"net.{i}.weight"], ck["encoder"][f"net.{i}.bias"]) for i in (0, 2, 4)]
        Dd = [(ck["decoder"][f"net.{i}.weight"], ck["decoder"][f"net.{i}.bias"]) for i in (0, 2, 4)]
        AE_SCALER = joblib.load("models/ae_scaler.pkl")
        _sc, _mn = AE_SCALER.scale_, AE_SCALER.min_
        def _mlp(h, L):
            for j, (W, b) in enumerate(L):
                h = h @ W.T + b
                if j < 2: h = np.tanh(h)
            return h
        def recon_mse(ae_in):
            x = (ae_in * _sc + _mn).astype(np.float32)
            r = _mlp(_mlp(x, E), Dd).astype(np.float64)
            return float(np.mean((ae_in - (r - _mn) / _sc)**2))
        print("[setup] AAE: numpy float32 forward pass")
elif args.ae == "standin":
    b = joblib.load("ae_sklearn_standin.joblib")
    def recon_mse(ae_in):
        x = b["scaler"].transform(ae_in.reshape(1, -1))
        return float(np.mean((ae_in - b["scaler"].inverse_transform(b["ae"].predict(x))[0])**2))
else:
    def recon_mse(ae_in): return 0.0

# ───────────── 5. controller ───────────────────────────────────────────────
CORR = {"n": 0}
def optim_controller(y_s, int_P, int_T, u_prev):
    A_s, I_s, R_s, P_s, T_s = y_s
    err_P, err_T = SETPT["P"] - P_s, SETPT["T_s"] - T_s
    CORR["n"] = CORR["n"] + 1 if (abs(err_P) < args.corr_tol and abs(err_T) < args.corr_tol) else 0
    in_corr = CORR["n"] >= args.corr_n
    s_raw = np.array([A_s, I_s, R_s, P_s, T_s, err_P, err_T, int_P / 1000.0, int_T / 1000.0])
    if args.clip_int:
        s_raw[7:9] = np.clip(s_raw[7:9], INT_LO, INT_HI)
    d_s = knn(tree_s, (s_raw - mu_x[:9]) / sd_x[:9])
    kap_s = {"lcb": args.kappa_lcb, "hybrid_s": ramp(d_s, S50, S99)}.get(args.mode, 0.0)
    if args.kappa_ss is not None and args.mode == "lcb" and in_corr:
        kap_s = args.kappa_ss

    def terms(u):
        sa = np.hstack([s_raw, u])
        mu, sd = critic(sa)
        kap = ramp(knn(tree_sa, (sa - mu_x) / sd_x), SA50, SA99) if args.mode == "hybrid_sa" else kap_s
        if args.mode == "trust":                      # report the effective kappa
            kap = args.kappa_trust * max(0.0, sd - SD_THR) / sd if sd > 0 else 0.0
        ae = recon_mse(np.array([A_s, I_s, R_s, P_s, T_s, err_P, err_T, u[0], u[1]]))
        return mu, sd, kap, ae

    def obj(u_vec):
        if args.u_box == "data":
            u_vec = np.clip(u_vec, U_LO, U_HI)
        mu, sd, kap, ae = terms(u_vec)
        du = u_vec - u_prev
        return args.w_ae * ae - args.w_q * (mu - kap * sd) + args.w_du * float(du @ du)

    if args.opt == "grid":
        if args.ae != "none" or args.mode in ("hybrid_sa", "trust"):
            raise SystemExit("--opt grid supports --ae none with modes xgb, mean, lcb, hybrid_s")
        g1, g2 = np.meshgrid(np.linspace(U_LO[0], U_HI[0], args.n_grid), np.linspace(U_LO[1], U_HI[1], args.n_grid))
        Ug = np.column_stack([g1.ravel(), g2.ravel()])
        Xg = np.hstack([np.tile(s_raw, (len(Ug), 1)), Ug])
        if args.mode == "xgb":
            mu_g, sd_g = Q_model.predict(Xg), np.zeros(len(Ug))
        else:
            mu_g, sd_g = CRIT.predict(Xg, gp_mean=not args.no_gp_mean)
        Jg = -args.w_q * (mu_g - kap_s * sd_g) + args.w_du * np.sum((Ug - u_prev) ** 2, 1)
        u0 = Ug[int(np.argmin(Jg))]
        r = minimize(obj, u0, method="Nelder-Mead", options={"maxiter": 100, "disp": False,
                                                             "initial_simplex": np.array([u0, u0 + [0.02, 0], u0 + [0, 0.002]])})
        u_ref = np.clip(r.x, U_LO, U_HI)
        u_vec = u_ref if obj(u_ref) < obj(u0) else u0
        int_P += err_P * DT; int_T += err_T * DT          # integrators are logged only; not used by a no-int critic
        mu, sd, kap, ae = terms(u_vec)
        d_sa = knn(tree_sa, (np.hstack([s_raw, u_vec]) - mu_x) / sd_x)
        return u_vec, int_P, int_T, dict(Q_mu=mu, Q_sd=sd, kappa=kap, ae_recon=ae,
                                         d_state=d_s, d_sa=d_sa, d_state_ood=d_s > S99, d_sa_ood=d_sa > SA99)

    bounds = list(zip(U_LO, U_HI))
    x0 = u_prev if np.any(u_prev) else np.array([0.7, 0.5])
    if args.u_box == "data":
        x0 = np.clip(x0, U_LO, U_HI)
    cands = []
    r = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 250, "disp": False})
    cands.append((r.x, np.clip(r.x, U_LO, U_HI)))
    r = minimize(obj, x0, method="Powell", bounds=bounds, options={"maxiter": 250, "disp": False})
    cands.append((r.x, np.clip(r.x, U_LO, U_HI)))
    if HAVE_DIRECT and not args.no_direct:
        r = direct(obj, bounds, maxiter=40, eps=1e-1)
        cands.append((r.x, np.clip(r.x, U_LO, U_HI)))
    objs = [obj(c[1]) for c in cands]
    u_raw, u_vec = cands[int(np.argmin(objs))]
    # anti-windup (raw vs clipped), as in the repo
    if u_vec[0] == u_raw[0]: int_P += err_P * DT
    if u_vec[1] == u_raw[1]: int_T += err_T * DT
    mu, sd, kap, ae = terms(u_vec)
    d_sa = knn(tree_sa, (np.hstack([s_raw, u_vec]) - mu_x) / sd_x)
    return u_vec, int_P, int_T, dict(Q_mu=mu, Q_sd=sd, kappa=kap, ae_recon=ae, in_corr=in_corr,
                                     d_state=d_s, d_sa=d_sa, d_state_ood=d_s > S99, d_sa_ood=d_sa > SA99)

# ───────────── 6. simulation ───────────────────────────────────────────────
logs = []
n_steps = int(args.hours * 3600 // DT)
for run in range(args.first_run, LAST + 1):
    rng = np.random.default_rng(args.seed_base + run)
    y_s, int_P, int_T, u_prev = Y0.copy(), 0.0, 0.0, np.zeros(2)
    CORR["n"] = 0
    prev_P = prev_T = None
    t, t0 = 0.0, time.time()
    for _ in range(n_steps):
        y_meas = y_s.copy()
        y_meas[0] += rng.normal(0.0, SIG_MONO_S)
        y_meas[3] += rng.normal(0.0, SIG_POLY_S)
        y_meas[4] += rng.normal(0.0, SIG_TEMP_S)
        _tc = time.perf_counter()
        u, int_P, int_T, diag = optim_controller(y_meas, int_P, int_T, u_prev)
        diag["solve_ms"] = 1e3 * (time.perf_counter() - _tc)
        sol = solve_ivp(lambda tt, yy: cstr_ode_scaled(tt, yy, u[0], u[1]), [0, DT], y_s,
                        method="BDF", atol=1e-8, rtol=1e-6)
        y_s = sol.y[:, -1]; t += DT
        reward = 0.0 if prev_P is None else compute_reward(y_meas[3], y_meas[4], prev_P, prev_T, SETPT)
        prev_P, prev_T, u_prev = y_meas[3], y_meas[4], u.copy()
        logs.append(dict(run_id=run, time_min=t / 60, A=y_meas[0], I=y_meas[1], R=y_meas[2],
                         P=y_meas[3], T=y_meas[4],
                         err_P=SETPT["P"] - y_meas[3], err_T=SETPT["T_s"] - y_meas[4],
                         int_err_P=int_P / 1000, int_err_T=int_T / 1000,
                         u_I=u[0], u_Tc=u[1], reward=reward, **diag))
    if not args.quiet:
        R = sum(l["reward"] for l in logs if l["run_id"] == run)
        print(f"[{NAME}] run {run}  {time.time()-t0:.0f}s  sum reward={R:.1f}", flush=True)
    pd.DataFrame(logs).to_csv(CSV_OUT, index=False)
print(f"✔ [{NAME}] saved {CSV_OUT}")
