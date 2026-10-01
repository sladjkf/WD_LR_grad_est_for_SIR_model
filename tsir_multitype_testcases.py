import csv
import json
import numpy as np
from scipy.stats import norm
from tsir_multitype import *
import os
import multiprocessing as mp

# =====================================================================
# GLOBAL CONFIGURATION & COMPUTATIONAL CONSTANTS
# =====================================================================
# Path to FSP results CSV file
FSP_CSV_PATH = "output/data/fsp_jacobians_results.csv" 

#ROW_INDEX = 0                              # Target row index in CSV to analyze (0-based)

ROW_INDEX = 4 # good case for beta: WD is bang on, actually onwards from here its pretty good

NUM_MC_DRAWS = 10000                         # Number of independent Monte Carlo draws (M)
EPS_FD_V = 1                              # Finite difference perturbation step size
EPS_FD_BETA = 0.1
EQUAL_EPS = True                           # Use equal perturbation step size across coordinates
CONF_LEVEL = 0.95                          # Confidence interval coverage (e.g., 0.95 for 95% CI)
MASTER_SEED = 42                           # Base seed for reproducible random streams

# Global configuration (can be overriden or set elsewhere in script)
CACHE_PATH = "output/data/tsir_multitype_test_cache"  # Set to None to disable caching

# =====================================================================
# 2. HELPER DATA LOADER
# =====================================================================

def load_fsp_csv_row(filepath, row_idx):
    """Loads and deserializes a target row from the FSP CSV results file."""
    with open(filepath, mode="r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if row_idx >= len(rows):
        raise IndexError(f"Requested row index {row_idx} exceeds total available CSV rows ({len(rows)}).")

    row = rows[row_idx]
    
    parsed_data = {
        "case_id": int(row["case_id"]),
        "K": int(row["K"]),
        "N": np.array(json.loads(row["N"]), dtype=float),
        "i0": np.array(json.loads(row["i0"]), dtype=float),
        "T": int(row["T"]),
        "W": np.array(json.loads(row["W"]), dtype=float),
        "V": np.array(json.loads(row["V"]), dtype=float),
        "beta": np.array(json.loads(row["beta"]), dtype=float),
        "cum_E_I_nom": np.array(json.loads(row["cum_E_I_nom"]), dtype=float),
        # Vaccination Jacobian Ground Truth
        "J_V": np.array(json.loads(row["J_V"]), dtype=float),
        "J_V_interval": np.array(json.loads(row["J_V_interval"]), dtype=float) if "J_V_interval" in row and row["J_V_interval"] else None,
        # Transmission Jacobian Ground Truth
        "J_beta": np.array(json.loads(row["J_beta"]), dtype=float),
        "J_beta_interval": np.array(json.loads(row["J_beta_interval"]), dtype=float) if "J_beta_interval" in row and row["J_beta_interval"] else None,
    }

    return parsed_data


# =====================================================================
# 3. ESTIMATOR COMPARISON DRIVER
# =====================================================================
def _compute_draw_worker(args):
    """
    Top-level worker function to evaluate gradient estimators for a single MC draw m
    across all location components k.
    """
    m, child_seed, N, i0, V, beta, W, T, K, eps_fd_v, eps_fd_beta, equal_eps = args

    sub_ss = child_seed.spawn(2)
    pop_seed = sub_ss[0].generate_state(1)[0]
    dyn_seed = sub_ss[1].generate_state(1)[0]

    fd_v_m = np.zeros((K, K))
    weak_v_m = np.zeros((K, K))
    fd_beta_m = np.zeros((K, K))
    weak_beta_m = np.zeros((K, K))

    for k in range(K):
        obj_fn_k = lambda traj, loc=k: np.sum(traj[:, K + loc])

        # Vaccination (V) Gradients
        fd_v_res = tsir_multipop_fd_v(
            N, i0, V, beta, W, T, obj_fn=obj_fn_k,
            pop_seed=pop_seed, dyn_seed=dyn_seed, eps=eps_fd_v, equal_eps=equal_eps
        )
        fd_v_m[k, :] = fd_v_res.flatten()

        weak_v_res = tsir_multipop_v(
            N, i0, V, beta, W, T, obj_fn=obj_fn_k,
            pop_seed=pop_seed, dyn_seed=dyn_seed
        )
        weak_v_m[k, :] = weak_v_res.flatten()

        # Transmission (Beta) Gradients
        fd_beta_res = tsir_multipop_fd_beta(
            N, i0, V, beta, W, T, obj_fn=obj_fn_k,
            pop_seed=pop_seed, dyn_seed=dyn_seed, eps=eps_fd_beta, equal_eps=equal_eps
        )
        fd_beta_m[k, :] = fd_beta_res.flatten()

        weak_beta_res = tsir_multipop_beta(
            N, i0, V, beta, W, T, obj_fn=obj_fn_k,
            pop_seed=pop_seed, dyn_seed=dyn_seed
        )
        weak_beta_m[k, :] = weak_beta_res.flatten()

    return m, fd_v_m, weak_v_m, fd_beta_m, weak_beta_m


def compare_estimators_to_fsp():
    """
    Evaluates nominal path expectations alongside finite-difference and weak-derivative
    gradient estimators for both V and beta against FSP ground truth.

    Parallelizes Monte Carlo draws using multiprocessing.Pool and supports JSON caching.
    """
    data = load_fsp_csv_row(FSP_CSV_PATH, ROW_INDEX)

    K = data["K"]
    N, i0, V, beta, W, T = data["N"], data["i0"], data["V"], data["beta"], data["W"], data["T"]
    cum_E_I_fsp = data["cum_E_I_nom"]
    J_V_fsp = data["J_V"]
    J_beta_fsp = data["J_beta"]

    eps_fd_v = getattr(globals(), 'EPS_FD_V', EPS_FD_V)
    eps_fd_beta = getattr(globals(), 'EPS_FD_BETA', EPS_FD_BETA)
    equal_eps = getattr(globals(), 'EQUAL_EPS', EQUAL_EPS)
    cache_path = getattr(globals(), 'CACHE_PATH', None)
    num_workers = getattr(globals(), 'NUM_WORKERS', os.cpu_count())

    print("=" * 140)
    print(f" EXPERIMENT CONFIGURATION (Case ID {data['case_id']}, K={K} Locations)")
    print("=" * 140)
    print(f"CSV Source File : {FSP_CSV_PATH} (Row {ROW_INDEX})")
    print(f"N               : {N.tolist()}")
    print(f"i0              : {i0.tolist()}")
    print(f"V               : {V.tolist()}")
    print(f"beta            : {beta.tolist()}")
    print(f"T               : {T}")
    print(f"MC Draws (M)    : {NUM_MC_DRAWS}")
    print(f"FD Step (eps_V) : {eps_fd_v}")
    print(f"FD Step (eps_b) : {eps_fd_beta}")
    print(f"Parallel Workers: {num_workers}")
    print(f"Cache Path      : {cache_path if cache_path is not None else 'Disabled (Fresh Run)'}")

    # --- Cache Initialization ---
    cache_file = None
    loaded_from_cache = False

    if cache_path is not None:
        os.makedirs(cache_path, exist_ok=True)
        cache_filename = (
            f"mc_cache_row{ROW_INDEX}_M{NUM_MC_DRAWS}_seed{MASTER_SEED}_"
            f"epsV{eps_fd_v}_epsB{eps_fd_beta}_eq{int(equal_eps)}.json"
        )
        cache_file = os.path.join(cache_path, cache_filename)

        if os.path.exists(cache_file):
            print(f"\n[CACHE] Loading pre-computed results from: {cache_file}")
            with open(cache_file, "r") as f:
                cached_data = json.load(f)

            mc_cum_E_I = np.array(cached_data["mc_cum_E_I"])
            mc_J_FD_V = np.array(cached_data["mc_J_FD_V"])
            mc_J_Weak_V = np.array(cached_data["mc_J_Weak_V"])
            mc_J_FD_Beta = np.array(cached_data["mc_J_FD_Beta"])
            mc_J_Weak_Beta = np.array(cached_data["mc_J_Weak_Beta"])
            loaded_from_cache = True

    # --- Parallel Monte Carlo Computation ---
    if not loaded_from_cache:
        mc_J_FD_V = np.zeros((NUM_MC_DRAWS, K, K))
        mc_J_Weak_V = np.zeros((NUM_MC_DRAWS, K, K))
        mc_J_FD_Beta = np.zeros((NUM_MC_DRAWS, K, K))
        mc_J_Weak_Beta = np.zeros((NUM_MC_DRAWS, K, K))

        master_ss = np.random.SeedSequence(MASTER_SEED)

        # 1. Nominal expectation simulation
        nom_ss = master_ss.spawn(1)[0].spawn(2)
        nom_pop_seed = nom_ss[0].generate_state(1)[0]
        nom_dyn_seed = nom_ss[1].generate_state(1)[0]

        obj_fn_full = lambda traj: np.sum(traj[:, K:], axis=0)
        mc_cum_E_I = tsir_multipop(
            N, i0, V, beta, W, T, obj_fn=obj_fn_full, num_sims=NUM_MC_DRAWS,
            pop_seed=nom_pop_seed, dyn_seed=nom_dyn_seed
        )

        # 2. Parallel outer loop across MC draws
        child_seeds = master_ss.spawn(NUM_MC_DRAWS)
        tasks = [
            (m, child_seeds[m], N, i0, V, beta, W, T, K, eps_fd_v, eps_fd_beta, equal_eps)
            for m in range(NUM_MC_DRAWS)
        ]

        print(f"\nRunning {NUM_MC_DRAWS} Monte Carlo draws in parallel ({num_workers} processes)...")

        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_compute_draw_worker, tasks)

        for m, fd_v_m, weak_v_m, fd_beta_m, weak_beta_m in results:
            mc_J_FD_V[m] = fd_v_m
            mc_J_Weak_V[m] = weak_v_m
            mc_J_FD_Beta[m] = fd_beta_m
            mc_J_Weak_Beta[m] = weak_beta_m

        # Save results to JSON cache
        if cache_file is not None:
            cache_payload = {
                "mc_cum_E_I": mc_cum_E_I.tolist(),
                "mc_J_FD_V": mc_J_FD_V.tolist(),
                "mc_J_Weak_V": mc_J_Weak_V.tolist(),
                "mc_J_FD_Beta": mc_J_FD_Beta.tolist(),
                "mc_J_Weak_Beta": mc_J_Weak_Beta.tolist(),
            }
            with open(cache_file, "w") as f:
                json.dump(cache_payload, f)
            print(f"[CACHE] Saved computation results to cache: {cache_file}")

    # --- Statistical Summary & Reporting ---
    z_alpha = norm.ppf(1.0 - (1.0 - CONF_LEVEL) / 2.0)

    # TABLE 1: NOMINAL CUMULATIVE EXPECTATION COMPARISON
    mean_nom = np.mean(mc_cum_E_I, axis=0)
    se_nom = np.std(mc_cum_E_I, axis=0, ddof=1) / np.sqrt(NUM_MC_DRAWS)
    err_nom = mean_nom - cum_E_I_fsp
    ci_nom_l, ci_nom_u = err_nom - z_alpha * se_nom, err_nom + z_alpha * se_nom

    print("\n" + "=" * 140)
    print(" TABLE 1: NOMINAL CUMULATIVE EXPECTATION COMPARISON")
    print("=" * 140)
    print(f"{'Loc (k)':<8} | {'FSP Ground Truth':<18} | {'Nominal MC Mean':<18} | {'Error (MC - FSP)':<18} | {f'{int(CONF_LEVEL*100)}% CI for Error':<25} | {'0 Excluded?'}")
    print("-" * 140)
    for k in range(K):
        b_away = (ci_nom_l[k] > 0) or (ci_nom_u[k] < 0)
        print(f"{k:<8} | {cum_E_I_fsp[k]:<18.6f} | {mean_nom[k]:<18.6f} | {err_nom[k]:<18.6f} | [{ci_nom_l[k]:10.6f}, {ci_nom_u[k]:10.6f}] | {'YES' if b_away else 'NO'}")

    def print_combined_jacobian_table(table_num, param_name, mc_fd, mc_weak, J_fsp):
        mean_fd = np.mean(mc_fd, axis=0)
        se_fd = np.std(mc_fd, axis=0, ddof=1) / np.sqrt(NUM_MC_DRAWS)
        err_fd = mean_fd - J_fsp
        ci_fd_l, ci_fd_u = err_fd - z_alpha * se_fd, err_fd + z_alpha * se_fd

        mean_weak = np.mean(mc_weak, axis=0)
        se_weak = np.std(mc_weak, axis=0, ddof=1) / np.sqrt(NUM_MC_DRAWS)
        err_weak = mean_weak - J_fsp
        ci_weak_l, ci_weak_u = err_weak - z_alpha * se_weak, err_weak + z_alpha * se_weak

        diff_fd_wd = mc_fd - mc_weak
        mean_diff = np.mean(diff_fd_wd, axis=0)
        se_diff = np.std(diff_fd_wd, axis=0, ddof=1) / np.sqrt(NUM_MC_DRAWS)
        ci_diff_l = mean_diff - z_alpha * se_diff
        ci_diff_u = mean_diff + z_alpha * se_diff

        print("\n" + "=" * 140)
        print(f" TABLE {table_num}: {param_name.upper()} JACOBIAN GRADIENT ESTIMATOR COMPARISON")
        print("=" * 140)
        print(f"{'Entry (k,j)':<11} | {'FSP Truth':<10} | {'FD Mean':<10} | {'WD Mean':<10} | {f'FD Err [{int(CONF_LEVEL*100)}% CI]':<25} | {f'WD Err [{int(CONF_LEVEL*100)}% CI]':<25} | {f'FD - WD Diff [{int(CONF_LEVEL*100)}% CI]':<26}")
        print("-" * 140)

        fd_bounded_away, wd_bounded_away, diff_bounded_away = 0, 0, 0

        for k in range(K):
            for j in range(K):
                is_fd_sig = (ci_fd_l[k, j] > 0) or (ci_fd_u[k, j] < 0)
                is_wd_sig = (ci_weak_l[k, j] > 0) or (ci_weak_u[k, j] < 0)
                is_diff_sig = (ci_diff_l[k, j] > 0) or (ci_diff_u[k, j] < 0)

                if is_fd_sig: fd_bounded_away += 1
                if is_wd_sig: wd_bounded_away += 1
                if is_diff_sig: diff_bounded_away += 1

                fd_ci_str = f"{err_fd[k, j]:.3f} [{ci_fd_l[k, j]:.3f}, {ci_fd_u[k, j]:.3f}]"
                wd_ci_str = f"{err_weak[k, j]:.3f} [{ci_weak_l[k, j]:.3f}, {ci_weak_u[k, j]:.3f}]"
                diff_ci_str = f"{mean_diff[k, j]:.3f} [{ci_diff_l[k, j]:.3f}, {ci_diff_u[k, j]:.3f}]"

                print(f"({k}, {j}):{'':<5} | {J_fsp[k, j]:<10.4f} | {mean_fd[k, j]:<10.4f} | {mean_weak[k, j]:<10.4f} | {fd_ci_str:<25} | {wd_ci_str:<25} | {diff_ci_str:<26}")

        print("-" * 140)
        print(f"Summary ({param_name}): FD vs FSP err != 0 in {fd_bounded_away}/{K*K} CIs; WD vs FSP err != 0 in {wd_bounded_away}/{K*K} CIs; FD vs WD diff != 0 in {diff_bounded_away}/{K*K} CIs.")

    # TABLE 2 & TABLE 3
    print_combined_jacobian_table(2, "Vaccination (V)", mc_J_FD_V, mc_J_Weak_V, J_V_fsp)
    print_combined_jacobian_table(3, "Transmission (Beta)", mc_J_FD_Beta, mc_J_Weak_Beta, J_beta_fsp)
    print("=" * 140 + "\n")


if __name__ == "__main__":
    compare_estimators_to_fsp()

if __name__ == "__main__":
    compare_estimators_to_fsp()