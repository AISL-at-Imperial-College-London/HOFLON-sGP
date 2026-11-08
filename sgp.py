"""
sgp.py — sparse (inducing-point) GP critic for HOFLON, NumPy/scikit-learn only.

Model
-----
  * ARD kernel, squared-exponential ("rbf") or Matern-5/2 ("matern52")
  * constant prior mean m (mean return, or a low quantile = "pessimistic")
  * Titsias / VFE posterior: with inducing inputs Z and hyperparameters fixed,
    the optimal q(u) is closed form and uses ALL N training points, O(N M^2)
  * hyperparameters (signal variance, lengthscales, noise) from an exact GP fit
    on a random subset of the training rows (scikit-learn), then frozen

Sparsity / accuracy controls (arguments of fit_critic)
------------------------------------------------------
  m_ind     number of inducing points M — THE sparsity knob. Memory O(N M),
            training O(N M^2), each prediction O(M) for the mean and O(M^2) for
            the variance. 200-1000 is the useful range here.
  inducing  how Z is chosen: "kmeans" (cluster centres, default) or "random"
            (random training rows).
  n_hyp     subset size for hyperparameter learning, cost O(n_hyp^3) per
            optimiser iteration. 1000-3000 is practical on a laptop.
  kernel    "rbf" or "matern52" (rougher functions, heavier tails).
  jitter    diagonal regulariser on K_uu (relative to signal variance).

predict() returns the latent mean and latent (epistemic) std of f(x) — no
observation noise — which is what a lower confidence bound on Q should use.
"""
import time
import numpy as np
from scipy.linalg import solve_triangular


class SparseGP:
    def __init__(self, Z, lengthscales, signal_var, noise_var, y_mean=0.0, y_std=1.0,
                 kernel="rbf", jitter=1e-6):
        self.Z = np.asarray(Z, dtype=np.float64)
        self.ell = np.asarray(lengthscales, dtype=np.float64)
        self.s2, self.sn2 = float(signal_var), float(noise_var)
        self.y_mean, self.y_std = float(y_mean), float(y_std)
        self.kernel, self.jitter = kernel, jitter

    # ---------------------------------------------------------------- kernel
    def _k(self, A, B):
        A_, B_ = A / self.ell, B / self.ell
        sq = np.maximum(np.sum(A_**2, 1)[:, None] + np.sum(B_**2, 1)[None, :] - 2.0 * A_ @ B_.T, 0.0)
        if self.kernel == "rbf":
            return self.s2 * np.exp(-0.5 * sq)
        r = np.sqrt(5.0 * sq)                                   # Matern 5/2
        return self.s2 * (1.0 + r + r**2 / 3.0) * np.exp(-r)

    # ------------------------------------------------------------------- fit
    def fit(self, X, y, chunk=5000):
        X = np.asarray(X, dtype=np.float64)
        yn = (np.asarray(y, dtype=np.float64) - self.y_mean) / self.y_std
        M = self.Z.shape[0]
        L = np.linalg.cholesky(self._k(self.Z, self.Z) + self.jitter * self.s2 * np.eye(M))
        AAT, Ay = np.zeros((M, M)), np.zeros(M)
        for i in range(0, X.shape[0], chunk):
            A = solve_triangular(L, self._k(self.Z, X[i:i + chunk]), lower=True) / np.sqrt(self.sn2)
            AAT += A @ A.T
            Ay += A @ yn[i:i + chunk]
        LB = np.linalg.cholesky(np.eye(M) + AAT)
        c = solve_triangular(LB, Ay, lower=True) / np.sqrt(self.sn2)
        Linv = solve_triangular(L, np.eye(M), lower=True)
        LBinv_Linv = solve_triangular(LB, Linv, lower=True)
        self.alpha = LBinv_Linv.T @ c                                # mean weights
        self.W = Linv.T @ Linv - LBinv_Linv.T @ LBinv_Linv           # K_uu^-1 - Sigma
        return self

    # --------------------------------------------------------------- predict
    def predict(self, X, return_std=True):
        """Mean and latent std, in ORIGINAL target units. X already input-scaled."""
        Kxu = self._k(np.atleast_2d(np.asarray(X, dtype=np.float64)), self.Z)
        mu = self.y_mean + self.y_std * (Kxu @ self.alpha)
        if not return_std:
            return mu
        var = self.s2 - np.sum((Kxu @ self.W) * Kxu, axis=1)       # BLAS; was einsum("ij,jk,ik->i")
        return mu, self.y_std * np.sqrt(np.maximum(var, 1e-12))

    @property
    def prior_std(self):
        return self.y_std * np.sqrt(self.s2)

    # ----------------------------------------------------------- persistence
    _KEYS = ["Z", "ell", "s2", "sn2", "y_mean", "y_std", "kernel", "jitter", "alpha", "W"]
    def state_dict(self):
        return {k: getattr(self, k) for k in self._KEYS}

    @classmethod
    def from_state_dict(cls, d):
        obj = cls(d["Z"], d["ell"], d["s2"], d["sn2"], d["y_mean"], d["y_std"],
                  d.get("kernel", "rbf"), d["jitter"])
        obj.alpha, obj.W = d["alpha"], d["W"]
        return obj


def fit_critic(Xs_tr, y_tr, prior="data", prior_q=0.01, n_hyp=1500, m_ind=500,
               kernel="rbf", inducing="kmeans", jitter=1e-6, seed=0, verbose=True):
    """Hyperparameters on a subset (exact GP) -> inducing points -> VFE posterior.
    Xs_tr must already be standardised. Returns (SparseGP, info dict)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(seed)
    D = Xs_tr.shape[1]
    y_std = float(np.std(y_tr))
    y_mean = float(np.mean(y_tr)) if prior == "data" else float(np.quantile(y_tr, prior_q))

    t0 = time.time()
    idx = rng.choice(len(y_tr), min(n_hyp, len(y_tr)), replace=False)
    base = (RBF(np.ones(D), (1e-2, 1e3)) if kernel == "rbf"
            else Matern(np.ones(D), (1e-2, 1e3), nu=2.5))
    k0 = ConstantKernel(1.0, (1e-2, 1e2)) * base + WhiteKernel(1e-2, (1e-6, 1.0))
    gpr = GaussianProcessRegressor(k0, normalize_y=False, random_state=seed)
    gpr.fit(Xs_tr[idx], (y_tr[idx] - y_mean) / y_std)
    k = gpr.kernel_
    s2, ell, sn2 = k.k1.k1.constant_value, np.atleast_1d(k.k1.k2.length_scale), k.k2.noise_level
    t_hyp = time.time() - t0

    t0 = time.time()
    if inducing == "kmeans":
        Z = MiniBatchKMeans(m_ind, random_state=seed, n_init=3, batch_size=4096).fit(Xs_tr).cluster_centers_
    else:
        Z = Xs_tr[rng.choice(len(Xs_tr), m_ind, replace=False)]
    gp = SparseGP(Z, ell, s2, sn2, y_mean, y_std, kernel, jitter).fit(Xs_tr, y_tr)
    info = dict(prior=prior, prior_mean=y_mean, kernel=kernel, inducing=inducing,
                m_ind=m_ind, n_hyp=int(len(idx)), signal_var=float(s2), noise_var=float(sn2),
                lengthscales=ell.tolist(), t_hyp_s=round(t_hyp, 1), t_fit_s=round(time.time() - t0, 1))
    if verbose:
        print(f"[sgp] prior mean {y_mean:.2f} ({prior}); kernel {kernel}; "
              f"hyp {t_hyp:.0f}s on {len(idx)} pts; posterior on N={len(y_tr)}, M={m_ind}")
    return gp, info
