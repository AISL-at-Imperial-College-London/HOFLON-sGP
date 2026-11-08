"""Builds tables/numbers.tex (LaTeX macros for every number in the paper) and tables/table_main.tex (Table I)
from tables/per_run.csv, the historical data, tables/landscape.json and the critic files."""
import numpy as np, pandas as pd, glob, re, os
from poly_metrics import historical, RES
M = pd.read_csv("tables/per_run.csv"); H = historical()
FAIL = 6000.0
M["fail"] = M.reward < FAIL
out = {}
def fmt(x, d=0):
    if x < 0 and round(x, d) != 0: return "$-$" + fmt(-x, d)
    s = f"{x:,.{d}f}"; return s.replace(",", "{,}")
def put(k, v): out[k] = v
# ---- historical
put("HistMean", fmt(H.reward.mean())); put("HistStd", fmt(H.reward.std())); put("HistWorst", fmt(H.reward.min())); put("HistBest", fmt(H.reward.max()))
put("HistTmax", fmt(H.Tmax.max(), 1)); put("HistMvTc", fmt(H.mv_Tc_late.median(), 3)); put("HistOod", fmt(100 * H.frac_ood_late.median(), 1))
fails = M[M.fail]; put("FailGapMax", fmt(fails.reward.max() if len(fails) else 0))
# ---- configurations for the main table
ROWS = [  # (cfg, label, group)
    ("xgbnoAE",        "none",                                  "x"),
    ("xgbAAE",         "AAE, $\\lambda_1=0.083$ (published, HOFLON)", "x"),
    ("xgbAAEw5000",    "AAE, $\\lambda_1\\times10^4$",          "x"),
    ("mtK0",           "$\\kappa=0$ (GP mean only)",            "m"),
    ("mtK3",           "$\\kappa=3$ (proposed)",                "m"),
    ("sgpK0",          "$\\kappa=0$ (GP mean only)",            "g"),
    ("sgpK0AAEw0.5",   "$\\kappa=0$ + AAE, $\\lambda_1=0.083$", "g"),
    ("sgpK0AAEw5000",  "$\\kappa=0$ + AAE, $\\lambda_1\\times10^4$", "g"),
    ("sgpK3",          "$\\kappa=3$",                           "g"),
]
def summ(d):
    return dict(n=len(d), crit=d.member.nunique(), fail=int(d.fail.sum()), med=d.reward.median(), worst=d.reward.min(),
                tmax=d.Tmax.max(), mv=d.mv_Tc_late.median(), mvI=d.mv_I_late.median(), poff=d.Poff_late.median(), toff=d.Toff_late.median(),
                ood=100 * d.frac_ood_late.median(), oode=100 * d.frac_ood_early.median(), ms=d.solve_ms.median() if "solve_ms" in d and d.solve_ms.notna().any() else np.nan)
MT = M[M.seed <= 13]                                   # main table: seeds 11-13, fully paired
S = {c: summ(MT[MT.cfg == c]) for c, _, _ in ROWS if (MT.cfg == c).any()}
SA = {c: summ(M[M.cfg == c]) for c in ["sgpK0", "sgpK3", "mtK3"] if (M.cfg == c).any()}   # all seeds 11-20
BOX = {"xgbnoAE": "2", "xgbAAE": "3", "xgbAAEw5000": "4", "mtK3": "5", "sgpK3": "6"}
lines = [r"\begin{tabular}{@{}c l c r r c c c c c c@{}}", r"\toprule",
         r"Fig.~\ref{fig:box} & Support term in the online objective & Failed & \multicolumn{1}{c}{Median} & \multicolumn{1}{c}{Worst} & Max.\ peak & Coolant & \multicolumn{2}{c}{Steady-state offset} & \multicolumn{2}{c}{Off-support [\%]} \\ \cmidrule(lr){8-9} \cmidrule(l){10-11}",
         r"box & & runs & \multicolumn{1}{c}{$R$} & \multicolumn{1}{c}{$R$} & $T$ [K] & move [K] & $P$ [kg\,m$^{-3}$] & $T$ [K] & 0--20\,h & $>$20\,h \\ \midrule"]
for c, lab, grp in ROWS:
    if grp == "x" and c == "xgbnoAE": lines.append(r" & \multicolumn{10}{@{}l}{\emph{HOFLON~\cite{durkin2026hoflon}: XGBoost critic $\hat Q$, objective~\eqref{eq:hoflon}}}\\")
    if grp == "m" and c == "mtK0": lines.append(r"\midrule & \multicolumn{10}{@{}l}{\emph{HOFLON-sGP: sparse-GP critic with Mat\'ern-5/2 kernel, objective~\eqref{eq:sgp}}}\\")
    if grp == "g" and c == "sgpK0": lines.append(r"\midrule & \multicolumn{10}{@{}l}{\emph{Sparse-GP critic with squared-exponential kernel, objective~\eqref{eq:sgp} with optional AAE term}}\\")
    if c not in S: continue
    s = S[c]; b = (lambda t: f"\\textbf{{{t}}}") if c == "mtK3" else (lambda t: t)
    ood = f"{s['ood']:.0f}" if s['ood'] >= 10 else f"{s['ood']:.1f}"
    oode = f"{s['oode']:.0f}" if s['oode'] >= 10 else f"{s['oode']:.1f}"
    lines.append(f"{BOX.get(c, '')} & {lab} & {b(str(s['fail']))}/{s['n']} & {b(fmt(s['med']))} & {b(fmt(s['worst']))} & {s['tmax']:.1f} & {s['mv']:.2f} & {s['poff']:.2f} & {s['toff']:.2f} & {oode} & {ood}\\\\")
lines.append(r"\midrule " + f"1 & Historical PI (100 logged runs) & {(H.reward < FAIL).sum()}/{len(H)} & {fmt(H.reward.median())} & {fmt(H.reward.min())} & {H.Tmax.max():.1f} & {H.mv_Tc_late.median():.2f} & {H.Poff_late.median():.2f} & {H.Toff_late.median():.2f} & {100*H.frac_ood_early.median():.0f} & {100*H.frac_ood_late.median():.0f}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("tables/table_main.tex", "w").write("\n".join(lines) + "\n")
# ---- macros for every configuration
def cname(c): return re.sub(r"[^A-Za-z]", "", c.replace("0.5", "half").replace("5000", "tenk").replace("500", "onek").replace("50", "hund").replace("3", "three").replace("0", "zero"))
for c, s in S.items():
    p = cname(c)
    put(p + "Fail", str(s["fail"])); put(p + "N", str(s["n"])); put(p + "Med", fmt(s["med"])); put(p + "Worst", fmt(s["worst"]))
    put(p + "Tmax", fmt(s["tmax"], 1)); put(p + "Mv", fmt(s["mv"], 2)); put(p + "Poff", fmt(s["poff"], 2)); put(p + "Toff", fmt(s["toff"], 1)); put(p + "Ood", fmt(s["ood"], 0 if s["ood"] >= 10 else 1)); put(p + "OodE", fmt(s["oode"], 0 if s["oode"] >= 10 else 1))
    if not np.isnan(s["ms"]): put(p + "Ms", fmt(s["ms"], 0))
for c, s in SA.items():
    p = cname(c) + "All"
    put(p + "Fail", str(s["fail"])); put(p + "N", str(s["n"])); put(p + "Med", fmt(s["med"])); put(p + "Worst", fmt(s["worst"]))
    put(p + "Tmax", fmt(s["tmax"], 1)); put(p + "Mv", fmt(s["mv"], 2)); put(p + "Poff", fmt(s["poff"], 2)); put(p + "Toff", fmt(s["toff"], 1))
with open("tables/numbers.tex", "w") as f:
    for k, v in out.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open("tables/table_main.tex").read()); print(len(out), "macros"); print(sorted(out)[:60])

# ================= extra macros used in the text =================
import subprocess, json
def q(c, f=lambda d: d): return f(M[M.cfg == c])
# held-out seeds for the proposed setting
d = M[(M.cfg == "sgpK3") & (M.seed >= 14)]
if len(d):
    put("KthreeHoN", str(len(d))); put("KthreeHoFail", str(int((d.reward < FAIL).sum())))
    put("KthreeHoMed", fmt(d.reward.median())); put("KthreeHoWorst", fmt(d.reward.min()))
d = M[(M.cfg == "sgpK3")]; put("KthreeAllBest", fmt(d.reward.max())); put("KthreeMvI", fmt(d.mv_I_late.median() * 1000, 0))
put("KthreeTlate", fmt(d.T_late.median(), 1)); put("KthreePlate", fmt(d.P_late.median(), 1))
# kappa regimes (seeds 11-13)
for c, p in [("sgpK0.25", "Kqtr"), ("sgpK1", "Kone"), ("sgpK1.5", "Konehalf"), ("sgpK2", "Ktwo"), ("sgpK2.5", "Ktwohalf"), ("sgpK4", "Kfour"), ("sgpK5", "Kfive"),
             ("sgpK0.5ss3", "Ksched"), ("sgpK0.5ss4", "Kschedfour"), ("sgpK0.5", "Khalfdev"), ("sgpK3", "Kthreedev")]:
    d = M[(M.cfg == c) & (M.seed <= 13)]
    if not len(d): continue
    put(p + "Med", fmt(d.reward.median())); put(p + "Worst", fmt(d.reward.min())); put(p + "Fail", str(int((d.reward < FAIL).sum())))
    put(p + "Poff", fmt(d.Poff_late.median(), 1)); put(p + "Mv", fmt(d.mv_Tc_late.median(), 2)); put(p + "Ood", fmt(100 * d.frac_ood_late.median(), 0 if d.frac_ood_late.median() >= 0.1 else 1))
    put(p + "Sd", fmt(d.sd_late_med.median(), 0))
# limit cycle of kappa = 0.5: coolant range in the late phase, and lag-1 autocorrelation
import glob as _g
RESD = "results/"
def late_actions(cfg):
    out = []
    for f in _g.glob(RESD + f"*_Z_{cfg}_f*"):
        for fs in _g.glob(f + "/*.csv"):
            x = pd.read_csv(fs); out.append(x[x.time_min > 3000].assign(src=fs))
    return pd.concat(out) if out else None
L = late_actions("sgpK0.5")
if L is not None:
    ac = []
    for (s, r), g in L.groupby(["src", "run_id"]):
        y = g.u_Tc.values - g.u_Tc.mean(); ac.append(np.corrcoef(y[:-1], y[1:])[0, 1])
    put("KhalfLagOne", fmt(np.median(ac), 2)); put("KhalfTcLo", fmt(280 + 120 * L.u_Tc.quantile(0.02), 1)); put("KhalfTcHi", fmt(280 + 120 * L.u_Tc.quantile(0.98), 1))
for cfg, p in [("sgpK3", "Kthree"), ("sgpK1.5", "Konehalf")]:
    L = late_actions(cfg)
    if L is not None:
        put(p + "uI", fmt(L.u_I.mean(), 2)); put(p + "Tc", fmt(280 + 120 * L.u_Tc.mean(), 1)); put(p + "fI", fmt(2.5 * L.u_I.mean(), 2))
Hl = pd.read_csv("data/multi_run_dataset.csv"); Hl = Hl[Hl.time_min > 3000]
put("PIuI", fmt(Hl.u_I.median(), 2)); put("PITc", fmt(280 + 120 * Hl.u_Tc.median(), 1)); put("PIfI", fmt(2.5 * Hl.u_I.median(), 2))
put("HistTlate", fmt(H.T_late.median(), 1))
put("BandLo", fmt(280 + 120 * 0.501, 1)); put("BandHi", fmt(280 + 120 * 0.537, 1))
# GP mean + AAE runaways
for c, p in [("sgpK0AAEw0.5", "GAAE"), ("sgpK0AAEw5000", "GAAEtenk"), ("sgpK0", "Gmean")]:
    d = M[M.cfg == c]
    if len(d):
        put(p + "Run", str(int((d.Tmax > 380).sum()))); put(p + "N", str(len(d))); put(p + "TmaxLo", fmt(d.Tmax.min(), 0)); put(p + "TmaxHi", fmt(d.Tmax.max(), 0))
        put(p + "Plate", fmt(d.P_late.median(), 0))
# AAE pairing statistics
from poly_aae_analysis import pair_stats
X5 = ["rs42", "rs0", "rs7", "rs1", "rs2"]; G5 = [f"f{i}" for i in range(5)]
P = pair_stats("xgbAAE", "xgbnoAE", X5, 0.5 / 6)
if len(P):
    put("XpairN", str(len(P))); put("XdivMed", fmt(P.first_div.median(), 0)); put("XdivMin", str(int(P.first_div.min()))); put("XdivMax", str(int(P.first_div.max())))
    put("XpenMed", f"{P.pen_med.median():.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
    put("XpenMax", f"{P.pen_max.max():.2f}")
    put("XsameOutcome", str(int((abs(P.RA - P.RB) < 1).sum())))
P = pair_stats("sgpK0AAEw0.5", "sgpK0", G5, 0.5 / 6)
if len(P): put("GdivMed", fmt(P.first_div.median(), 0))
# solve times
# runs executed without concurrent jobs: original paper runs (seeds 11-13) and Matern critics m3, m4
for c, p in [("xgbAAE", "xgbAAE"), ("sgpK3", "sgpKthree")]:
    d = M[(M.cfg == c) & (M.seed <= 13)]
    if len(d) and d.solve_ms.notna().any(): put(p + "MsMed", fmt(d.solve_ms.median(), 0))
with open("tables/numbers.tex", "w") as f:
    for k, v in out.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(out), "macros total")

# landscape scales (Fig. 1)
import json as _js; [put(k, v) for k, v in _js.load(open("tables/landscape.json")).items()] if os.path.exists("tables/landscape.json") else None
with open("tables/numbers.tex", "w") as f:
    for k, v in out.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
# ---- XGBoost ablation specifics (seeds 11-13)
X = MT[MT.cfg.isin(["xgbAAE", "xgbnoAE"])]
if (X.cfg == "xgbnoAE").any():
    n = X[X.cfg == "xgbnoAE"]; a = X[X.cfg == "xgbAAE"]
    put("XnoAEcrit", str(n[n.reward < FAIL].member.nunique())); put("XnoAEhot", str(int((n.Tmax > 360).sum()))); put("XAAEhot", str(int((a.Tmax > 360).sum())))
    put("XnoAETmed", fmt(n.Tmax.median(), 1)); put("XAAETmed", fmt(a.Tmax.median(), 1)); put("KthreeTmed", fmt(MT[MT.cfg == "sgpK3"].Tmax.median(), 1))
    w = X.pivot_table(index=["member", "seed"], columns="cfg", values="reward")
    put("XAAEbetter", str(int((w.xgbAAE > w.xgbnoAE).sum()))); put("XpairsN", str(len(w)))
    put("XrsSevenNo", fmt(n[n.member == "rs7"].reward.median())); put("XrsSevenAAE", fmt(a[a.member == "rs7"].reward.median()))
    put("XrsZeroAAEtwelve", fmt(float(w.loc[("rs0", 12), "xgbAAE"]))); put("XrsZeroNotwelve", fmt(float(w.loc[("rs0", 12), "xgbnoAE"])))
    from scipy.stats import mannwhitneyu
    put("MWp", f"{mannwhitneyu(MT[MT.cfg=='sgpK3'].reward, a.reward).pvalue:.2f}")
W = MT[MT.cfg == "xgbAAEw5000"]
if len(W): put("XtenkN", str(len(W)))
with open("tables/numbers.tex", "w") as f:
    for k, v in out.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- failure type of the XGBoost runs without AAE (seeds 11-13)
n = MT[(MT.cfg == "xgbnoAE") & (MT.reward < FAIL)]
if len(n):
    put("XfailPlo", fmt(n.P_late.min(), 0)); put("XfailPhi", fmt(n.P_late.max(), 0))
    put("XfailTlo", fmt(n.Tmax.min(), 0)); put("XfailThi", fmt(n.Tmax.max(), 0))
# ---- HOFLON-sGP with the Matern-5/2 critics (five fits; seeds 11-13 and held-out 14-20)
import joblib as _jl
d = M[(M.cfg == "mtK3") & (M.seed >= 14)]
if len(d):
    put("MtHoN", str(len(d))); put("MtHoFail", str(int((d.reward < FAIL).sum())))
    put("MtHoMed", fmt(d.reward.median())); put("MtHoWorst", fmt(d.reward.min()))
d = M[M.cfg == "mtK3"]
if len(d):
    put("MtAllBest", fmt(d.reward.max())); put("MtAllWorst", fmt(d.reward.min()))
    t = d[d.seed <= 13]; put("MtTmed", fmt(t.Tmax.median(), 1))
for c, p in [("mtK0.5", "MtKhalf"), ("mtK1", "MtKone"), ("mtK2", "MtKtwo"), ("mtK3", "MtKthree"), ("mtK0", "MtKzero")]:
    d = M[(M.cfg == c) & (M.seed <= 13)]
    if not len(d): continue
    put(p + "Med", fmt(d.reward.median())); put(p + "Worst", fmt(d.reward.min())); put(p + "Fail", str(int((d.reward < FAIL).sum())))
    put(p + "Poff", fmt(d.Poff_late.median(), 2)); put(p + "Toff", fmt(d.Toff_late.median(), 2)); put(p + "Mv", fmt(d.mv_Tc_late.median(), 2))
    put(p + "Ood", fmt(100 * d.frac_ood_late.median(), 1)); put(p + "OodMax", fmt(100 * d.frac_ood_late.max(), 1))
    put(p + "OodE", fmt(100 * d.frac_ood_early.median(), 0)); put(p + "Tmax", fmt(d.Tmax.max(), 1))
    put(p + "TmaxLo", fmt(d.Tmax.min(), 0)); put(p + "TmaxHi", fmt(d.Tmax.max(), 0)); put(p + "Plate", fmt(d.P_late.median(), 0))
    put(p + "TcLo", fmt(d.Tc_min_late.min(), 1)); put(p + "TcHi", fmt(d.Tc_max_late.max(), 1))
d = M[M.cfg.isin(["mtK0.5", "mtK1", "mtK2", "mtK3"]) & (M.seed <= 13)]
if len(d):
    g = d.groupby("cfg")
    put("MtSweepFail", str(int((d.reward < FAIL).sum()))); put("MtSweepN", str(len(d)))
    put("MtSweepMedLo", fmt(g.reward.median().min())); put("MtSweepMedHi", fmt(g.reward.median().max()))
    put("MtSweepOodMax", fmt(100 * d.frac_ood_late.max(), 1)); put("MtSweepMvMax", fmt(g.mv_Tc_late.median().max(), 2))
    put("MtSweepTcLo", fmt(d.Tc_min_late.min(), 1)); put("MtSweepTcHi", fmt(d.Tc_max_late.max(), 1))
L = []
for f in _g.glob(RESD + "*_Z_mtK3_m*"):
    for fs in _g.glob(f + "/*.csv"):
        x = pd.read_csv(fs); L.append(x[x.time_min > 3000])
if L:
    L = pd.concat(L); put("MtKthreefI", fmt(2.5 * L.u_I.mean(), 2)); put("MtKthreeTc", fmt(280 + 120 * L.u_Tc.mean(), 1))
d = M[M.cfg == "mtK3"]
d = d[d.member.isin(["m3", "m4"])]
if len(d) and d.solve_ms.notna().any(): put("mtKthreeMsMed", fmt(d.solve_ms.median(), 0))
d = M[(M.cfg == "xgbAAE") & (M.seed >= 14)]
if len(d):
    put("XHoN", str(len(d))); put("XHoFail", str(int((d.reward < FAIL).sum()))); put("XHoMed", fmt(d.reward.median())); put("XHoWorst", fmt(d.reward.min()))
    a = M[M.cfg == "xgbAAE"]; put("XAllN", str(len(a))); put("XAllMed", fmt(a.reward.median())); put("XAllWorst", fmt(a.reward.min())); put("XAllTmax", fmt(a.Tmax.max(), 1))
Lz = []
for f in _g.glob(RESD + "*_Z_mtK0_m*"):
    for fs in _g.glob(f + "/*.csv"):
        x = pd.read_csv(fs)
        for r, gr in x.groupby("run_id"):
            hit = gr[gr.u_Tc > 0.999]
            Lz.append(dict(hrs=hit.time_min.min() / 60 if len(hit) else np.nan, late_bound=bool((gr[gr.time_min > 3000].u_Tc > 0.999).all()),
                           sd_late=gr[gr.time_min > 3000].Q_sd.median()))
if Lz:
    Lz = pd.DataFrame(Lz); put("MtZeroHrs", fmt(Lz.hrs.median(), 0)); put("MtZeroBound", str(int(Lz.late_bound.sum()))); put("MtZeroBoundN", str(len(Lz)))
    put("MtZeroSdLate", fmt(Lz.sd_late.median(), 0))
inf = []
for s_ in range(5):
    fn = f"models/q_sgp_mt52_sd{s_}.joblib"
    if os.path.exists(fn): inf.append(_jl.load(fn)["info"])
if inf:
    r2 = [i["r2_heldout"] for i in inf]; cv = [i["cov95_heldout"] for i in inf]
    put("MtRtwoLo", f"{min(r2):.3f}"); put("MtRtwoHi", f"{max(r2):.3f}")
    put("MtCovLo", f"{100*min(cv):.0f}"); put("MtCovHi", f"{100*max(cv):.0f}")
    put("MtSdMed", fmt(np.median([i["sd_med_heldout"] for i in inf]), 1)); put("MtSigf", fmt(np.median([i["sigma_f"] for i in inf]), 0))
    put("MtNfits", str(len(inf)))
    rng = lambda lo, hi: lo if lo == hi else f"{lo}--{hi}"
    put("MtRtwo", rng(f"{min(r2):.3f}", f"{max(r2):.3f}")); put("MtCov", rng(f"{100*min(cv):.0f}", f"{100*max(cv):.0f}"))
z = M[(M.cfg == "mtK0")]
ok = z[z.reward >= FAIL]; bad = z[z.reward < FAIL]
if len(ok): put("MtZeroOkN", str(len(ok))); put("MtZeroOkCrit", str(ok.member.nunique())); put("MtZeroOkTlo", fmt(ok.Tmax.min(), 0)); put("MtZeroOkThi", fmt(ok.Tmax.max(), 0))
if len(bad): put("MtZeroBadTmax", fmt(bad.Tmax.max(), 0)); put("MtZeroBadTmin", fmt(bad.Tmax.min(), 0))
with open("tables/numbers.tex", "w") as f:
    for k, v in out.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(out), "macros after additions")
