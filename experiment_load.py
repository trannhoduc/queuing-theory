import os
import numpy as np
import matplotlib.pyplot as plt

from simulation import NetworkSimulator
from PARAMETERS import *


# ============================================================
# Helper to safely get parameters from PARAMETERS.py
# ============================================================

def get_param(name, default):
    return globals()[name] if name in globals() else default


# ============================================================
# Experiment configuration
# ============================================================

# Service rates
MU_R = get_param("MU_R", 5000.0)
MU_B = get_param("MU_B", 10000.0)

# Baseline offered traffic ratio
# Keeps the same URLLC/eMBB proportion while total load changes
LAMBDA_URLLC_BASE = get_param("LAMBDA_URLLC", 200.0)
LAMBDA_EMBB_BASE = get_param("LAMBDA_EMBB", 3000.0)

TOTAL_BASE = LAMBDA_URLLC_BASE + LAMBDA_EMBB_BASE
URLLC_RATIO = LAMBDA_URLLC_BASE / TOTAL_BASE
EMBB_RATIO = LAMBDA_EMBB_BASE / TOTAL_BASE

# Simulation controls
SIM_TIME = get_param("T_SIM", 2000.0)
WARMUP = get_param("WARMUP", 200.0)
BUFFER_SIZE_NODE1 = get_param("BUFFER_SIZE_NODE1", 100)
URLLC_DELAY_THRESHOLD = get_param("URLLC_DELAY_THRESHOLD", 1e-3)

# Load sweep
# Here, "load" means offered load at Node 1:
# rho = (lambda_urllc + lambda_embb) / mu_r
LOAD_VALUES = np.linspace(0.2, 0.9, 8)

# Number of independent runs for each load
NUM_REPETITIONS = 10

# Output folder
OUTPUT_DIR = "results_load_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Run one simulation for a given load and seed
# ============================================================

def run_one_simulation(rho, seed, buffer_size_node1):
    lambda_total = rho * MU_R
    lambda_urllc = URLLC_RATIO * lambda_total
    lambda_embb = EMBB_RATIO * lambda_total

    sim = NetworkSimulator(
        lambda_urllc=lambda_urllc,
        lambda_embb=lambda_embb,
        mu_r=MU_R,
        mu_b=MU_B,
        sim_time=SIM_TIME,
        warmup=WARMUP,
        delay_threshold_urllc=URLLC_DELAY_THRESHOLD,
        buffer_size_node1=buffer_size_node1,
        seed=seed,
    )

    results = sim.run()

    # End-to-end delay samples
    urlcc_samples = sim.stats.delay_samples["URLLC"]
    embb_samples = sim.stats.delay_samples["eMBB"]

    # Percentiles
    p95_urlcc = np.percentile(urlcc_samples, 95) if len(urlcc_samples) > 0 else np.nan
    p95_embb = np.percentile(embb_samples, 95) if len(embb_samples) > 0 else np.nan

    return {
        "rho": rho,
        "lambda_urllc": lambda_urllc,
        "lambda_embb": lambda_embb,
        "urlcc_p95_delay": p95_urlcc,
        "embb_p95_delay": p95_embb,
        "urlcc_loss_prob": results["URLLC"]["losses"] / results["URLLC"]["arrivals"]
        if results["URLLC"]["arrivals"] > 0 else np.nan,
        "embb_loss_prob": results["eMBB"]["loss_prob"],
        "urlcc_mean_delay": results["URLLC"]["mean_delay_total"],
        "embb_mean_delay": results["eMBB"]["mean_delay_total"],
        "urlcc_violation_prob_1ms": results["URLLC"]["delay_violation_prob_1ms"],
    }


# ============================================================
# Run load sweep with multiple seeds
# ============================================================

def run_load_sweep(buffer_size_node1, case_name):
    summary_rows = []

    avg_urlcc_p95 = []
    avg_embb_p95 = []
    avg_urlcc_loss = []
    avg_embb_loss = []
    avg_urlcc_mean_delay = []
    avg_embb_mean_delay = []
    avg_urlcc_violation = []

    std_urlcc_p95 = []
    std_embb_p95 = []
    std_urlcc_loss = []
    std_embb_loss = []

    for rho in LOAD_VALUES:
        run_results = []

        for seed in range(NUM_REPETITIONS):
            result = run_one_simulation(
                rho=rho,
                seed=seed,
                buffer_size_node1=buffer_size_node1
            )
            run_results.append(result)

        urlcc_p95_vals = np.array([r["urlcc_p95_delay"] for r in run_results], dtype=float)
        embb_p95_vals = np.array([r["embb_p95_delay"] for r in run_results], dtype=float)

        urlcc_loss_vals = np.array([r["urlcc_loss_prob"] for r in run_results], dtype=float)
        embb_loss_vals = np.array([r["embb_loss_prob"] for r in run_results], dtype=float)

        urlcc_mean_delay_vals = np.array([r["urlcc_mean_delay"] for r in run_results], dtype=float)
        embb_mean_delay_vals = np.array([r["embb_mean_delay"] for r in run_results], dtype=float)

        urlcc_violation_vals = np.array([r["urlcc_violation_prob_1ms"] for r in run_results], dtype=float)

        avg_urlcc_p95.append(np.nanmean(urlcc_p95_vals))
        avg_embb_p95.append(np.nanmean(embb_p95_vals))

        avg_urlcc_loss.append(np.nanmean(urlcc_loss_vals))
        avg_embb_loss.append(np.nanmean(embb_loss_vals))

        avg_urlcc_mean_delay.append(np.nanmean(urlcc_mean_delay_vals))
        avg_embb_mean_delay.append(np.nanmean(embb_mean_delay_vals))

        avg_urlcc_violation.append(np.nanmean(urlcc_violation_vals))

        std_urlcc_p95.append(np.nanstd(urlcc_p95_vals, ddof=1) if len(urlcc_p95_vals) > 1 else 0.0)
        std_embb_p95.append(np.nanstd(embb_p95_vals, ddof=1) if len(embb_p95_vals) > 1 else 0.0)

        std_urlcc_loss.append(np.nanstd(urlcc_loss_vals, ddof=1) if len(urlcc_loss_vals) > 1 else 0.0)
        std_embb_loss.append(np.nanstd(embb_loss_vals, ddof=1) if len(embb_loss_vals) > 1 else 0.0)

        summary_rows.append({
            "rho": rho,
            "lambda_urllc": URLLC_RATIO * rho * MU_R,
            "lambda_embb": EMBB_RATIO * rho * MU_R,
            "urlcc_p95_mean": avg_urlcc_p95[-1],
            "embb_p95_mean": avg_embb_p95[-1],
            "urlcc_loss_mean": avg_urlcc_loss[-1],
            "embb_loss_mean": avg_embb_loss[-1],
            "urlcc_mean_delay": avg_urlcc_mean_delay[-1],
            "embb_mean_delay": avg_embb_mean_delay[-1],
            "urlcc_violation_prob_1ms": avg_urlcc_violation[-1],
        })

    return {
        "case_name": case_name,
        "rho": np.array(LOAD_VALUES, dtype=float),

        "urlcc_p95_mean": np.array(avg_urlcc_p95, dtype=float),
        "embb_p95_mean": np.array(avg_embb_p95, dtype=float),
        "urlcc_loss_mean": np.array(avg_urlcc_loss, dtype=float),
        "embb_loss_mean": np.array(avg_embb_loss, dtype=float),
        "urlcc_mean_delay": np.array(avg_urlcc_mean_delay, dtype=float),
        "embb_mean_delay": np.array(avg_embb_mean_delay, dtype=float),
        "urlcc_violation_prob_1ms": np.array(avg_urlcc_violation, dtype=float),

        "urlcc_p95_std": np.array(std_urlcc_p95, dtype=float),
        "embb_p95_std": np.array(std_embb_p95, dtype=float),
        "urlcc_loss_std": np.array(std_urlcc_loss, dtype=float),
        "embb_loss_std": np.array(std_embb_loss, dtype=float),

        "rows": summary_rows,
    }


# ============================================================
# Save results table to CSV
# ============================================================

def save_csv(results_dict, filename):
    header = (
        "rho,lambda_urllc,lambda_embb,"
        "urlcc_p95_mean,embb_p95_mean,"
        "urlcc_loss_mean,embb_loss_mean,"
        "urlcc_mean_delay,embb_mean_delay,"
        "urlcc_violation_prob_1ms\n"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(header)
        for row in results_dict["rows"]:
            f.write(
                f"{row['rho']},"
                f"{row['lambda_urllc']},"
                f"{row['lambda_embb']},"
                f"{row['urlcc_p95_mean']},"
                f"{row['embb_p95_mean']},"
                f"{row['urlcc_loss_mean']},"
                f"{row['embb_loss_mean']},"
                f"{row['urlcc_mean_delay']},"
                f"{row['embb_mean_delay']},"
                f"{row['urlcc_violation_prob_1ms']}\n"
            )


# ============================================================
# Plot functions
# ============================================================

def plot_p95_delays(results_dict, filename_png):
    rho = results_dict["rho"]
    urlcc_p95_ms = 1000.0 * results_dict["urlcc_p95_mean"]
    embb_p95_ms = 1000.0 * results_dict["embb_p95_mean"]

    plt.figure(figsize=(8, 5))
    plt.plot(rho, urlcc_p95_ms, marker="o", label="URLLC")
    plt.plot(rho, embb_p95_ms, marker="s", label="eMBB")
    plt.xlabel("Load at Node 1, $\\rho = (\\lambda_U + \\lambda_E)/\\mu_R$")
    plt.ylabel("95th-percentile end-to-end delay (ms)")
    plt.title(f"95th-percentile delays vs load ({results_dict['case_name']})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename_png, dpi=300)
    plt.show()


def plot_loss_probabilities(results_dict, filename_png):
    rho = results_dict["rho"]
    urlcc_loss = results_dict["urlcc_loss_mean"]
    embb_loss = results_dict["embb_loss_mean"]

    plt.figure(figsize=(8, 5))
    plt.plot(rho, urlcc_loss, marker="o", label="URLLC")
    plt.plot(rho, embb_loss, marker="s", label="eMBB")
    plt.xlabel("Load at Node 1, $\\rho = (\\lambda_U + \\lambda_E)/\\mu_R$")
    plt.ylabel("Loss probability")
    plt.title(f"Loss probabilities vs load ({results_dict['case_name']})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename_png, dpi=300)
    plt.show()


def plot_urlcc_violation(results_dict, filename_png):
    rho = results_dict["rho"]
    violation = results_dict["urlcc_violation_prob_1ms"]

    plt.figure(figsize=(8, 5))
    plt.plot(rho, violation, marker="o")
    plt.xlabel("Load at Node 1, $\\rho = (\\lambda_U + \\lambda_E)/\\mu_R$")
    plt.ylabel("URLLC delay violation probability")
    plt.title(f"URLLC delay violation probability vs load ({results_dict['case_name']})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename_png, dpi=300)
    plt.show()


# ============================================================
# Pretty print summary
# ============================================================

def print_summary(results_dict):
    print("\n" + "=" * 90)
    print(f"SUMMARY: {results_dict['case_name']}")
    print("=" * 90)
    print(
        f"{'rho':>8} {'p95_URLLC(ms)':>15} {'p95_eMBB(ms)':>15} "
        f"{'loss_URLLC':>15} {'loss_eMBB':>15} {'Pvio_URLLC':>15}"
    )

    for i, rho in enumerate(results_dict["rho"]):
        print(
            f"{rho:8.3f} "
            f"{1000.0 * results_dict['urlcc_p95_mean'][i]:15.6f} "
            f"{1000.0 * results_dict['embb_p95_mean'][i]:15.6f} "
            f"{results_dict['urlcc_loss_mean'][i]:15.6e} "
            f"{results_dict['embb_loss_mean'][i]:15.6e} "
            f"{results_dict['urlcc_violation_prob_1ms'][i]:15.6e}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Infinite-buffer case
    results_infinite = run_load_sweep(
        buffer_size_node1=None,
        case_name="Infinite buffer"
    )

    print_summary(results_infinite)
    save_csv(results_infinite, os.path.join(OUTPUT_DIR, "load_sweep_infinite.csv"))
    plot_p95_delays(results_infinite, os.path.join(OUTPUT_DIR, "p95_delay_infinite.png"))
    plot_loss_probabilities(results_infinite, os.path.join(OUTPUT_DIR, "loss_prob_infinite.png"))
    plot_urlcc_violation(results_infinite, os.path.join(OUTPUT_DIR, "urlcc_violation_infinite.png"))

    # Finite-buffer case
    results_finite = run_load_sweep(
        buffer_size_node1=BUFFER_SIZE_NODE1,
        case_name=f"Finite buffer (B={BUFFER_SIZE_NODE1})"
    )

    print_summary(results_finite)
    save_csv(results_finite, os.path.join(OUTPUT_DIR, "load_sweep_finite.csv"))
    plot_p95_delays(results_finite, os.path.join(OUTPUT_DIR, "p95_delay_finite.png"))
    plot_loss_probabilities(results_finite, os.path.join(OUTPUT_DIR, "loss_prob_finite.png"))
    plot_urlcc_violation(results_finite, os.path.join(OUTPUT_DIR, "urlcc_violation_finite.png"))