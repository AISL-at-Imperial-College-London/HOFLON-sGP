#!/bin/bash
# Every closed-loop run in the paper. Run from the repository root. Each line appends one CSV to results/<name>/.
# Seeds: --seed_base 11 --first_run 0 --runs 3 -> seeds 11-13; --first_run 3 --runs 7 -> held-out seeds 14-20.
set -e
C="python poly_hoflon_sgp_cntrl.py --quiet"

# ---- HOFLON-sGP: Matern-5/2 critics (models/q_sgp_mt52_sd{0..4}.joblib), kappa = 3 fixed in advance
for s in 0 1 2 3 4; do
  G="--mode lcb --ae none --seed_base 11 --gp_file models/q_sgp_mt52_sd$s.joblib"
  $C $G --kappa_lcb 3 --first_run 0 --runs 3 --tag _Z_mtK3_m$s
  $C $G --kappa_lcb 3 --first_run 3 --runs 7 --tag _Z_mtK3_m$s
  $C $G --kappa_lcb 0 --first_run 0 --runs 3 --tag _Z_mtK0_m$s
  for k in 0.5 1 2; do $C $G --kappa_lcb $k --first_run 0 --runs 3 --tag _Z_mtK${k}_m$s; done
done

# ---- HOFLON (XGBoost + AAE) on the held-out seeds 14-20
for rs in rs42 rs0 rs7 rs1 rs2; do
  $C --mode xgb --xgb_file models/q_xgb_$rs.joblib --seed_base 11 --first_run 3 --runs 7 --tag _Z_xgbAAE_$rs
done

# ---- XGBoost rows, squared-exponential rows and kappa sweep (seeds 11-13 unless stated).
# The squared-exponential critics models/q_gpt_sgpr_data_noint_sd{0..4}.joblib are not included;
# create them with: python poly_learn_Q_sgp_gpytorch.py --no_int --iters 200 --seed $s --tag _sd$s  (torch, gpytorch)
$C --mode lcb --kappa_lcb 0 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK0_f0   # sgpK0_f0_s11-13
$C --mode lcb --kappa_lcb 0 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK0_f1   # sgpK0_f1_s11-13
$C --mode lcb --kappa_lcb 0 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK0_f2   # sgpK0_f2_s11-13
$C --mode lcb --kappa_lcb 0 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK0_f3   # sgpK0_f3_s11-13
$C --mode lcb --kappa_lcb 0 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK0_f4   # sgpK0_f4_s11-13
$C --mode lcb --kappa_lcb 0.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5_f0   # sgpK0.5_f0_s11-13
$C --mode lcb --kappa_lcb 0.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5_f1   # sgpK0.5_f1_s11-13
$C --mode lcb --kappa_lcb 0.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5_f2   # sgpK0.5_f2_s11-13
$C --mode lcb --kappa_lcb 0.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5_f3   # sgpK0.5_f3_s11-13
$C --mode lcb --kappa_lcb 0.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5_f4   # sgpK0.5_f4_s11-13
$C --mode lcb --kappa_lcb 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK3_f0   # sgpK3_f0_s11-13
$C --mode lcb --kappa_lcb 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK3_f1   # sgpK3_f1_s11-13
$C --mode lcb --kappa_lcb 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK3_f2   # sgpK3_f2_s11-13
$C --mode lcb --kappa_lcb 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK3_f3   # sgpK3_f3_s11-13
$C --mode lcb --kappa_lcb 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK3_f4   # sgpK3_f4_s11-13
$C --mode lcb --kappa_lcb 4 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK4_f0   # sgpK4_f0_s11-13
$C --mode lcb --kappa_lcb 4 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK4_f1   # sgpK4_f1_s11-13
$C --mode lcb --kappa_lcb 4 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK4_f2   # sgpK4_f2_s11-13
$C --mode lcb --kappa_lcb 4 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK4_f3   # sgpK4_f3_s11-13
$C --mode lcb --kappa_lcb 4 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK4_f4   # sgpK4_f4_s11-13
$C --mode lcb --kappa_lcb 1 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK1_f0   # sgpK1_f0_s11-13
$C --mode lcb --kappa_lcb 1 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK1_f1   # sgpK1_f1_s11-13
$C --mode lcb --kappa_lcb 1 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK1_f2   # sgpK1_f2_s11-13
$C --mode lcb --kappa_lcb 1 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK1_f3   # sgpK1_f3_s11-13
$C --mode lcb --kappa_lcb 1 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK1_f4   # sgpK1_f4_s11-13
$C --mode lcb --kappa_lcb 1.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK1.5_f0   # sgpK1.5_f0_s11-13
$C --mode lcb --kappa_lcb 1.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK1.5_f1   # sgpK1.5_f1_s11-13
$C --mode lcb --kappa_lcb 1.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK1.5_f2   # sgpK1.5_f2_s11-13
$C --mode lcb --kappa_lcb 1.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK1.5_f3   # sgpK1.5_f3_s11-13
$C --mode lcb --kappa_lcb 1.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK1.5_f4   # sgpK1.5_f4_s11-13
$C --mode lcb --kappa_lcb 2 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK2_f0   # sgpK2_f0_s11-13
$C --mode lcb --kappa_lcb 2 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK2_f1   # sgpK2_f1_s11-13
$C --mode lcb --kappa_lcb 2 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK2_f2   # sgpK2_f2_s11-13
$C --mode lcb --kappa_lcb 2 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK2_f3   # sgpK2_f3_s11-13
$C --mode lcb --kappa_lcb 2 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK2_f4   # sgpK2_f4_s11-13
$C --mode lcb --kappa_lcb 2.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK2.5_f0   # sgpK2.5_f0_s11-13
$C --mode lcb --kappa_lcb 2.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK2.5_f1   # sgpK2.5_f1_s11-13
$C --mode lcb --kappa_lcb 2.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK2.5_f2   # sgpK2.5_f2_s11-13
$C --mode lcb --kappa_lcb 2.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK2.5_f3   # sgpK2.5_f3_s11-13
$C --mode lcb --kappa_lcb 2.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK2.5_f4   # sgpK2.5_f4_s11-13
$C --mode lcb --kappa_lcb 0.25 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.25_f0   # sgpK0.25_f0_s11-13
$C --mode lcb --kappa_lcb 0.25 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.25_f1   # sgpK0.25_f1_s11-13
$C --mode lcb --kappa_lcb 0.25 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.25_f2   # sgpK0.25_f2_s11-13
$C --mode lcb --kappa_lcb 0.25 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.25_f3   # sgpK0.25_f3_s11-13
$C --mode lcb --kappa_lcb 0.25 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.25_f4   # sgpK0.25_f4_s11-13
$C --mode lcb --kappa_lcb 5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK5_f0   # sgpK5_f0_s11-13
$C --mode lcb --kappa_lcb 5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK5_f1   # sgpK5_f1_s11-13
$C --mode lcb --kappa_lcb 5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK5_f2   # sgpK5_f2_s11-13
$C --mode lcb --kappa_lcb 5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK5_f3   # sgpK5_f3_s11-13
$C --mode lcb --kappa_lcb 5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK5_f4   # sgpK5_f4_s11-13
$C --mode lcb --kappa_lcb 0.5 --kappa_ss 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5ss3_f0   # sgpK0.5ss3_f0_s11-13
$C --mode lcb --kappa_lcb 0.5 --kappa_ss 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5ss3_f1   # sgpK0.5ss3_f1_s11-13
$C --mode lcb --kappa_lcb 0.5 --kappa_ss 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5ss3_f2   # sgpK0.5ss3_f2_s11-13
$C --mode lcb --kappa_lcb 0.5 --kappa_ss 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5ss3_f3   # sgpK0.5ss3_f3_s11-13
$C --mode lcb --kappa_lcb 0.5 --kappa_ss 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5ss3_f4   # sgpK0.5ss3_f4_s11-13
$C --mode lcb --kappa_lcb 0.5 --kappa_ss 4 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5ss4_f0   # sgpK0.5ss4_f0_s11-13
$C --mode lcb --kappa_lcb 0.5 --kappa_ss 4 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5ss4_f1   # sgpK0.5ss4_f1_s11-13
$C --mode lcb --kappa_lcb 0.5 --kappa_ss 4 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5ss4_f2   # sgpK0.5ss4_f2_s11-13
$C --mode lcb --kappa_lcb 0.5 --kappa_ss 4 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5ss4_f3   # sgpK0.5ss4_f3_s11-13
$C --mode lcb --kappa_lcb 0.5 --kappa_ss 4 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK0.5ss4_f4   # sgpK0.5ss4_f4_s11-13
$C --mode lcb --kappa_lcb 0 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 3 --runs 7 --tag _Z_sgpK0_f0   # sgpK0_f0_s14-20
$C --mode lcb --kappa_lcb 0 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 3 --runs 7 --tag _Z_sgpK0_f1   # sgpK0_f1_s14-20
$C --mode lcb --kappa_lcb 0 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 3 --runs 7 --tag _Z_sgpK0_f2   # sgpK0_f2_s14-20
$C --mode lcb --kappa_lcb 0 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 3 --runs 7 --tag _Z_sgpK0_f3   # sgpK0_f3_s14-20
$C --mode lcb --kappa_lcb 0 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 3 --runs 7 --tag _Z_sgpK0_f4   # sgpK0_f4_s14-20
$C --mode lcb --kappa_lcb 0.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 3 --runs 7 --tag _Z_sgpK0.5_f0   # sgpK0.5_f0_s14-20
$C --mode lcb --kappa_lcb 0.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 3 --runs 7 --tag _Z_sgpK0.5_f1   # sgpK0.5_f1_s14-20
$C --mode lcb --kappa_lcb 0.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 3 --runs 7 --tag _Z_sgpK0.5_f2   # sgpK0.5_f2_s14-20
$C --mode lcb --kappa_lcb 0.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 3 --runs 7 --tag _Z_sgpK0.5_f3   # sgpK0.5_f3_s14-20
$C --mode lcb --kappa_lcb 0.5 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 3 --runs 7 --tag _Z_sgpK0.5_f4   # sgpK0.5_f4_s14-20
$C --mode lcb --kappa_lcb 0 --ae aae --w_ae 0.5 --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK0AAEw0.5_f0   # sgpK0AAEw0.5_f0_s11-13
$C --mode lcb --kappa_lcb 0 --ae aae --w_ae 0.5 --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK0AAEw0.5_f1   # sgpK0AAEw0.5_f1_s11-13
$C --mode lcb --kappa_lcb 0 --ae aae --w_ae 0.5 --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK0AAEw0.5_f2   # sgpK0AAEw0.5_f2_s11-13
$C --mode lcb --kappa_lcb 0 --ae aae --w_ae 0.5 --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK0AAEw0.5_f3   # sgpK0AAEw0.5_f3_s11-13
$C --mode lcb --kappa_lcb 0 --ae aae --w_ae 0.5 --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK0AAEw0.5_f4   # sgpK0AAEw0.5_f4_s11-13
$C --mode lcb --kappa_lcb 0 --ae aae --w_ae 5000 --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 0 --runs 3 --tag _Z_sgpK0AAEw5000_f0   # sgpK0AAEw5000_f0_s11-13
$C --mode lcb --kappa_lcb 0 --ae aae --w_ae 5000 --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 0 --runs 3 --tag _Z_sgpK0AAEw5000_f1   # sgpK0AAEw5000_f1_s11-13
$C --mode lcb --kappa_lcb 0 --ae aae --w_ae 5000 --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 0 --runs 3 --tag _Z_sgpK0AAEw5000_f2   # sgpK0AAEw5000_f2_s11-13
$C --mode lcb --kappa_lcb 0 --ae aae --w_ae 5000 --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 0 --runs 3 --tag _Z_sgpK0AAEw5000_f3   # sgpK0AAEw5000_f3_s11-13
$C --mode lcb --kappa_lcb 0 --ae aae --w_ae 5000 --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 0 --runs 3 --tag _Z_sgpK0AAEw5000_f4   # sgpK0AAEw5000_f4_s11-13
$C --mode lcb --kappa_lcb 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd0.joblib --first_run 3 --runs 7 --tag _Z_sgpK3_f0   # sgpK3_f0_s14-20
$C --mode lcb --kappa_lcb 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd1.joblib --first_run 3 --runs 7 --tag _Z_sgpK3_f1   # sgpK3_f1_s14-20
$C --mode lcb --kappa_lcb 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd2.joblib --first_run 3 --runs 7 --tag _Z_sgpK3_f2   # sgpK3_f2_s14-20
$C --mode lcb --kappa_lcb 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd3.joblib --first_run 3 --runs 7 --tag _Z_sgpK3_f3   # sgpK3_f3_s14-20
$C --mode lcb --kappa_lcb 3 --ae none --seed_base 11 --gp_file models/q_gpt_sgpr_data_noint_sd4.joblib --first_run 3 --runs 7 --tag _Z_sgpK3_f4   # sgpK3_f4_s14-20
$C --mode xgb --xgb_file models/q_xgb_rs42.joblib --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbAAE_rs42   # xgbAAE_rs42_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs42.joblib --ae none --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbnoAE_rs42   # xgbnoAE_rs42_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs0.joblib --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbAAE_rs0   # xgbAAE_rs0_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs0.joblib --ae none --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbnoAE_rs0   # xgbnoAE_rs0_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs7.joblib --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbAAE_rs7   # xgbAAE_rs7_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs7.joblib --ae none --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbnoAE_rs7   # xgbnoAE_rs7_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs1.joblib --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbAAE_rs1   # xgbAAE_rs1_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs1.joblib --ae none --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbnoAE_rs1   # xgbnoAE_rs1_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs2.joblib --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbAAE_rs2   # xgbAAE_rs2_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs2.joblib --ae none --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbnoAE_rs2   # xgbnoAE_rs2_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs42.joblib --w_ae 5000 --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbAAEw5000_rs42   # xgbAAEw5000_rs42_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs0.joblib --w_ae 5000 --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbAAEw5000_rs0   # xgbAAEw5000_rs0_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs7.joblib --w_ae 5000 --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbAAEw5000_rs7   # xgbAAEw5000_rs7_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs1.joblib --w_ae 5000 --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbAAEw5000_rs1   # xgbAAEw5000_rs1_s11-13
$C --mode xgb --xgb_file models/q_xgb_rs2.joblib --w_ae 5000 --seed_base 11 --first_run 0 --runs 3 --tag _Z_xgbAAEw5000_rs2   # xgbAAEw5000_rs2_s11-13
