"""
learn_Q_gpytorch.py — sparse-GP Q-critic trained with GPyTorch (polymer reactor).  Hyperparameters AND inducing-point locations are learned jointly on
all training rows:

  --method sgpr   Titsias / VFE collapsed bound, full batch
                  (gpytorch.kernels.InducingPointKernel + ExactMarginalLogLikelihood)
  --method svgp   Hensman et al. stochastic variational GP, minibatches
                  (VariationalStrategy + VariationalELBO); for large N or GPU

Targets and features are those of the XGBoost scripts (discounted return,
gamma 0.9); split by run (0-94 train, 95-99 held out); inputs standardised.

Export
------
The learned (kernel hyperparameters, noise, inducing points) are passed to
sgp.SparseGP, which computes the optimal VFE posterior q(u) on all training
rows in NumPy and saves it in the same .joblib format as the scikit-learn
trainers.  The controllers therefore read it with --gp_file and do not need
GPyTorch at run time (NumPy prediction of one point is much faster than a torch
call, which matters inside Nelder-Mead/Powell).  The exported posterior is the
exact closed-form VFE posterior; GPyTorch's own large-N prediction uses iterative
(CG) solves, so the two means differ slightly — both held-out scores are
reported.

    python learn_Q_gpytorch.py                         # -> polymer_reactor/q_gpt_sgpr_data.joblib
    python learn_Q_gpytorch.py --prior pessimistic     # -> polymer_reactor/q_gpt_sgpr_pessimistic.joblib
    python learn_Q_gpytorch.py --method svgp --m_ind 300 --epochs 50

Requires: torch, gpytorch (pip install gpytorch).
"""
import argparse, copy, json, os, sys, time
import numpy as np, pandas as pd, joblib
import torch, gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import r2_score, mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sgp import SparseGP  # noqa: E402

CASES = {
    "poly": dict(folder="polymer_reactor", csv="multi_run_dataset.csv",
                 run="run_id", order=["run_id", "time_min"],
                 cols=["A", "I", "R", "P", "T", "err_P", "err_T", "int_err_P", "int_err_T",
                       "u_I", "u_Tc"]),
}

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--case", default="poly", choices=list(CASES))
ap.add_argument("--method", default="sgpr", choices=["sgpr", "svgp"])
# model
ap.add_argument("--prior", default="data", choices=["data", "pessimistic"],
                help="constant prior mean: mean return, or the --prior_q quantile")
ap.add_argument("--prior_q", type=float, default=0.01)
ap.add_argument("--kernel", default="rbf", choices=["rbf", "matern52"])
ap.add_argument("--no_ard", action="store_true", help="one shared lengthscale instead of ARD")
ap.add_argument("--ls_init", type=float, default=1.0, help="initial lengthscale (standardised units)")
ap.add_argument("--ls_bounds", type=float, nargs=2, default=[1e-2, 1e3])
ap.add_argument("--noise_init", type=float, default=0.1, help="initial noise variance (normalised target)")
ap.add_argument("--noise_min", type=float, default=1e-4)
# sparsity
ap.add_argument("--m_ind", type=int, default=500, help="number of inducing points")
ap.add_argument("--init", default="kmeans", choices=["kmeans", "random"], help="inducing-point initialisation")
ap.add_argument("--fix_inducing", action="store_true", help="keep inducing points at their initial locations")
# optimisation
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--iters", type=int, default=400, help="sgpr: full-batch Adam iterations")
ap.add_argument("--epochs", type=int, default=40, help="svgp: passes over the data")
ap.add_argument("--batch", type=int, default=1024, help="svgp: minibatch size")
ap.add_argument("--restarts", type=int, default=1,
                help="random restarts (restart 0 uses --ls_init; others draw log-uniform lengthscales in [0.3, 10])")
ap.add_argument("--log_every", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--float32", action="store_true", help="default is float64 (more stable)")
ap.add_argument("--no_int", action="store_true", help="leave the integrator features out of the critic")
ap.add_argument("--tag", default="")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dtype = torch.float32 if args.float32 else torch.float64
torch.set_default_dtype(dtype)
dev = torch.device(args.device)

# ── data ──────────────────────────────────────────────────────────────────
S = CASES[args.case]; base = os.path.join(HERE, S["folder"])
df = pd.read_csv(os.path.join("data", S["csv"])).sort_values(S["order"]).reset_index(drop=True)
G = np.zeros(len(df))
for _, grp in df.groupby(S["run"], sort=False):
    g = 0.0
    for idx in grp.index[::-1]:
        g = df.at[idx, "reward"] + 0.9 * g
        G[idx] = g
COLS = [c for c in S["cols"] if not (args.no_int and c.startswith("int_"))]; D = len(COLS)
X, y = df[COLS].values.astype(np.float64), G
tr = df[S["run"]].values < 95; te = ~tr
scaler = StandardScaler().fit(X[tr]); Xs = scaler.transform(X)
y_std = float(y[tr].std())
m0 = float(y[tr].mean()) if args.prior == "data" else float(np.quantile(y[tr], args.prior_q))
yn = (y - m0) / y_std
Xtr = torch.tensor(Xs[tr], device=dev); ytr = torch.tensor(yn[tr], device=dev)
Xte = torch.tensor(Xs[te], device=dev)
N = Xtr.shape[0]
print(f"[{args.case}] N_train={N}, N_heldout={int(te.sum())}, D={D}, prior mean={m0:.2f} ({args.prior})")

# ── model pieces ──────────────────────────────────────────────────────────
def make_kernel(ls0):
    lc = gpytorch.constraints.Interval(*args.ls_bounds)
    ard = None if args.no_ard else D
    k = (gpytorch.kernels.RBFKernel(ard_num_dims=ard, lengthscale_constraint=lc) if args.kernel == "rbf"
         else gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard, lengthscale_constraint=lc))
    k.lengthscale = torch.as_tensor(ls0).reshape(1, -1).expand(1, ard or 1).clone()
    sk = gpytorch.kernels.ScaleKernel(k); sk.outputscale = 1.0
    return sk

def make_lik():
    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(args.noise_min))
    lik.noise = max(args.noise_init, 2 * args.noise_min)
    return lik

def init_Z(seed):
    if args.init == "kmeans":
        Z = MiniBatchKMeans(args.m_ind, random_state=seed, n_init=3, batch_size=4096).fit(Xs[tr]).cluster_centers_
    else:
        Z = Xs[tr][np.random.default_rng(seed).choice(N, args.m_ind, replace=False)]
    return torch.tensor(Z, device=dev)

class SGPR(gpytorch.models.ExactGP):
    def __init__(self, X, y, lik, Z, kern):
        super().__init__(X, y, lik)
        self.mean_module = gpytorch.means.ZeroMean()          # prior mean handled by centring y on m0
        self.covar_module = gpytorch.kernels.InducingPointKernel(kern, inducing_points=Z, likelihood=lik)
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, Z, kern):
        vd = gpytorch.variational.CholeskyVariationalDistribution(Z.size(0))
        vs = gpytorch.variational.VariationalStrategy(self, Z, vd,
                                                      learn_inducing_locations=not args.fix_inducing)
        super().__init__(vs)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kern
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

def scale_kernel(model):
    return model.covar_module.base_kernel if args.method == "sgpr" else model.covar_module

def inducing(model):
    return (model.covar_module.inducing_points if args.method == "sgpr"
            else model.variational_strategy.inducing_points)

def hyp_summary(model, lik):
    sk = scale_kernel(model)
    ls = sk.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    return f"s2={sk.outputscale.item():.3f} noise={lik.noise.item():.4f} ls[min/med/max]={ls.min():.2f}/{np.median(ls):.2f}/{ls.max():.2f}"

# ── training ──────────────────────────────────────────────────────────────
def train_once(r):
    rng = np.random.default_rng(args.seed + r)
    ls0 = (np.full(D if not args.no_ard else 1, args.ls_init) if r == 0
           else np.exp(rng.uniform(np.log(0.3), np.log(10.0), D if not args.no_ard else 1)))
    kern, lik, Z = make_kernel(ls0), make_lik(), init_Z(args.seed + r)
    if args.method == "sgpr":
        model = SGPR(Xtr, ytr, lik, Z, kern).to(dev)
        if args.fix_inducing:
            model.covar_module.inducing_points.requires_grad_(False)
        params = {id(p): p for p in list(model.parameters()) + list(lik.parameters()) if p.requires_grad}
        opt = torch.optim.Adam(params.values(), lr=args.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        model.train(); lik.train()
        for it in range(1, args.iters + 1):
            opt.zero_grad()
            loss = -mll(model(Xtr), ytr)
            loss.backward(); opt.step()
            if it % args.log_every == 0 or it == args.iters:
                print(f"  restart {r} iter {it:4d}  -bound/N={loss.item():.4f}  {hyp_summary(model, lik)}", flush=True)
        final = loss.item()
    else:
        model = SVGP(Z, kern).to(dev); lik = lik.to(dev)
        params = {id(p): p for p in list(model.parameters()) + list(lik.parameters()) if p.requires_grad}
        opt = torch.optim.Adam(params.values(), lr=args.lr)
        mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=N)
        model.train(); lik.train()
        for ep in range(1, args.epochs + 1):
            perm, tot = torch.randperm(N, device=dev), 0.0
            for i in range(0, N, args.batch):
                b = perm[i:i + args.batch]
                opt.zero_grad()
                loss = -mll(model(Xtr[b]), ytr[b])
                loss.backward(); opt.step()
                tot += loss.item() * len(b)
            if ep % max(1, args.log_every // 10) == 0 or ep == args.epochs:
                print(f"  restart {r} epoch {ep:3d}  -ELBO/N={tot / N:.4f}  {hyp_summary(model, lik)}", flush=True)
        final = tot / N
    return final, model, lik

t0 = time.time()
best = None
for r in range(args.restarts):
    final, model, lik = train_once(r)
    print(f"[restart {r}] final loss {final:.4f}")
    if best is None or final < best[0]:
        best = (final, model, lik, r)
final, model, lik, r_best = best
t_train = time.time() - t0

# ── extract hyperparameters + inducing points, build NumPy posterior ─────
sk = scale_kernel(model)
s2 = float(sk.outputscale.item())
ell = sk.base_kernel.lengthscale.detach().cpu().numpy().ravel().astype(np.float64)
if ell.size == 1: ell = np.full(D, ell[0])
sn2 = float(lik.noise.item())
Z = inducing(model).detach().cpu().numpy().astype(np.float64)
gp = SparseGP(Z, ell, s2, sn2, m0, y_std, kernel=args.kernel).fit(Xs[tr], y[tr])

# ── checks: GPyTorch vs exported NumPy model on held-out runs ────────────
model.eval(); lik.eval()
mu_t = []
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for i in range(0, Xte.shape[0], 2000):
        mu_t.append(model(Xte[i:i + 2000]).mean.cpu().numpy())
mu_t = m0 + y_std * np.concatenate(mu_t)
mu_np, sd_np = gp.predict(Xs[te])

def sc(a, b): return dict(R2=float(r2_score(a, b)), RMSE=float(np.sqrt(mean_squared_error(a, b))))
z = (y[te] - mu_np) / np.sqrt(sd_np**2 + sn2 * y_std**2)
rep = dict(case=args.case, method=args.method, prior=args.prior, prior_mean=m0, kernel=args.kernel,
           ard=not args.no_ard, m_ind=args.m_ind, init=args.init, learned_inducing=not args.fix_inducing,
           restarts=args.restarts, best_restart=r_best, final_loss=final, train_time_s=round(t_train, 1),
           signal_var=s2, noise_var=sn2,
           gpytorch_heldout_runs=sc(y[te], mu_t),
           exported_heldout_runs=sc(y[te], mu_np),
           exported_train=sc(y[tr], gp.predict(Xs[tr], return_std=False)),
           max_abs_mean_diff_gpytorch_vs_exported=float(np.max(np.abs(mu_t - mu_np))),
           exported_cov95_heldout=float(np.mean(np.abs(z) < 1.96)),
           exported_latent_sd_heldout_median=float(np.median(sd_np)),
           prior_sd=float(gp.prior_std),
           lengthscales=dict(zip(COLS, np.round(ell, 3).tolist())))
print(json.dumps({k: v for k, v in rep.items() if k != "lengthscales"}, indent=1))
print("lengthscales:", rep["lengthscales"])
if rep["exported_heldout_runs"]["R2"] < rep["gpytorch_heldout_runs"]["R2"] - 0.005:
    print("! exported model scores clearly below GPyTorch's own prediction — check before use")

name = f"q_gpt_{args.method}_{args.prior}{'_noint' if args.no_int else ''}{args.tag}"
json.dump(rep, open(os.path.join("models", name + "_report.json"), "w"), indent=2)
joblib.dump({"gp": gp.state_dict(), "scaler": scaler, "cols": COLS, "info": rep},
            os.path.join("models", name + ".joblib"))
print(f"✔ saved {S['folder']}/{name}.joblib  (use --gp_file {name}.joblib, or --gp_prefix q_gpt_{args.method} in run_experiments.py)")
