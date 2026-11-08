"""
critics.py — one loader for every critic file, so the controller and the
checks treat them the same way.

    crit = load_critic("q_resgp.joblib")
    mu, sd = crit.predict(X)        # X: raw features (n, 11) in COLS order; sd is None for XGBoost

Supported files
  q_discounted.joblib      plain XGBoost regressor (repo)                 kind "xgb"
  q_gpt_sgpr_*.joblib      sparse GP on standardised inputs               kind "gp"
  q_resgp*.joblib          XGBoost mean + sparse GP on its residuals      kind "xgb_gp"
"""
import os, sys
import numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgp import SparseGP  # noqa: E402

COLS = ["A", "I", "R", "P", "T", "err_P", "err_T", "int_err_P", "int_err_T", "u_I", "u_Tc"]


class Critic:
    def __init__(self, kind, xgb=None, gp=None, scaler=None, info=None, path="", cols=None):
        self.kind, self.xgb, self.gp, self.scaler = kind, xgb, gp, scaler
        # columns the model was trained on (subset of COLS); inputs are always given in COLS order
        self.idx = [COLS.index(c) for c in (cols or COLS)]
        self.info, self.path = (info or {}), path

    @property
    def has_sd(self):
        return self.gp is not None

    @property
    def sd_quantiles(self):
        """latent sd on held-out runs (in-support reference), if the trainer stored it"""
        return self.info.get("sd_heldout_quantiles", {})

    def predict(self, X, gp_mean=True):
        """gp_mean=False (xgb_gp only): mean = XGBoost alone, sd from the GP"""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))[:, self.idx]
        if self.kind == "xgb":
            return self.xgb.predict(X.astype(np.float32)).astype(np.float64), None
        mu, sd = self.gp.predict((X - self.scaler.mean_) / self.scaler.scale_)   # = scaler.transform, no validation overhead
        if self.kind == "xgb_gp":
            m = self.xgb.predict(X.astype(np.float32)).astype(np.float64)
            mu = m + mu if gp_mean else m
        return mu, sd


def load_critic(path):
    obj = joblib.load(path)
    if not isinstance(obj, dict):                       # bare XGBoost model
        return Critic("xgb", xgb=obj, path=path)
    kind = obj.get("kind", "gp")
    if kind == "xgb":                                   # XGBoost saved with its column list
        return Critic("xgb", xgb=obj["xgb"], path=path, cols=obj.get("cols"))
    gp = SparseGP.from_state_dict(obj["gp"])
    return Critic(kind, xgb=obj.get("xgb"), gp=gp, scaler=obj["scaler"], info=obj.get("info"), path=path,
                  cols=obj.get("cols"))
