import os
import numpy as np
import matplotlib.pyplot as plt

from simulation import NetworkSimulator
from PARAMETERS import *


def get_param(name, default):
    return globals()[name] if name in globals() else default


# ============================================================
# Fixed parameters
# ============================================================

MU_R = get_param("MU_R", 5000.0)
MU_B = get_param("MU_B", 10000.0)

LAMBDA_EMBB_FIXED = 3000.0
URLLC_RATES = np.arange(50.0, 401.0, 50.0)

SIM_TIME = get_param("T_SIM", 2000.0)
WARMUP = get_param("WARMUP", 200.0)
BUFFER_SIZE_NODE1 = get_param("BUFFER_SIZE_NODE1", 100)
URLLC_DELAY_THRESHOLD = get_param("URLLC_DELAY_THRESHOLD", 1e-3)

NUM_REPETITIONS = 10

OUTPUT_DIR = "results_vary_urllc"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_one_simulation(lambda_urllc, lambda_embb, mu_r, mu_b, seed, buffer_size_node1):
    sim = NetworkSimulator(
        lambda_urllc=lambda_urllc,
        lambda_embb=lambda_embb,
        mu_r=mu_r,
        mu_b=mu_b,
        sim_time=SIM_TIME,
        warmup=WARMUP,
        delay_threshold_urllc=URLLC_DELAY_THRESHOLD,
        buffer_size_node1=buffer_size_node1,
        seed=seed,
    )

    results = sim.run()

    urlcc_samples = sim.stats.delay_samples["URLLC"]
    embb_samples = sim.stats.delay_samples["eMBB"]

    p95_urlcc = np.percentile(urlcc_samples, 95) if len(urlcc_samples) > 0 else np.nan
    p95_embb = np.percentile(embb_samples, 95) if len(embb_samples) > 0 else np.nan

    return {
        "lambda_urllc": lambda_urllc,
        "lambda_embb": lambda_embb,
        "urlcc_p95_delay": p95_urlcc,
        "embb_p95_delay": p95_embb,
        "urlcc_loss_prob": results["URLLC"]["losses"] / results["URLLC"]["arrivals"]
        if results["URLLC"]["arrivals"] > 0 else np.nan,
        "embb_loss_prob": results["eMBB"]["loss_prob"],
        "urlcc_mean_delay": results["URLLC"]["mean_delay_total"],
        "embb_mean_delay": results["eMBB"]["mean_delay_total"],
        "urlcc_mean_wait_node1": results["URLLC"]["mean_wait_node1"],
        "embb_mean_wait_node1": results["eMBB"]["mean_wait_node1"],
        "urlcc_violation_prob_1ms": results["URLLC"]["delay_violation_prob_1ms"],
    }


def run_sweep(buffer_size_node1, case_name):
    rows = []

    avg_urlcc_p95 = []
    avg_embb_p95 = []
    avg_urlcc_loss = []
    avg_embb_loss = []
    avg_urlcc_mean_delay = []
    avg_embb_mean_delay = []
    avg_urlcc_mean_wait_n1 = []
    avg_embb_mean_wait_n1 = []
    avg_urlcc_violation = []

    for lambda_urllc in URLLC_RATES:
        run_results = []

        for seed in range(NUM_REPETITIONS):
            r = run_one_simulation(
                lambda_urllc=lambda_urllc,
                lambda_embb=LAMBDA_EMBB_FIXED,
                mu_r=MU_R,
                mu_b=MU_B,
                seed=seed,
                buffer_size_node1=buffer_size_node1,
            )
            run_results.append(r)

        urlcc_p95_vals = np.array([r["urlcc_p95_delay"] for r in run_results], dtype=float)
        embb_p95_vals = np.array([r["embb_p95_delay"] for r in run_results], dtype=float)

        urlcc_loss_vals = np.array([r["urlcc_loss_prob"] for r in run_results], dtype=float)
        embb_loss_vals = np.array([r["embb_loss_prob"] for r in run_results], dtype=float)

        urlcc_mean_delay_vals = np.array([r["urlcc_mean_delay"] for r in run_results], dtype=float)
        embb_mean_delay_vals = np.array([r["embb_mean_delay"] for r in run_results], dtype=float)

        urlcc_mean_wait_vals = np.array([r["urlcc_mean_wait_node1"] for r in run_results], dtype=float)
        embb_mean_wait_vals = np.array([r["embb_mean_wait_node1"] for r in run_results], dtype=float)

        urlcc_violation_vals = np.array([r["urlcc_violation_prob_1ms"] for r in run_results], dtype=float)

        avg_urlcc_p95.append(np.nanmean(urlcc_p95_vals))
        avg_embb_p95.append(np.nanmean(embb_p95_vals))
        avg_urlcc_loss.append(np.nanmean(urlcc_loss_vals))
        avg_embb_loss.append(np.nanmean(embb_loss_vals))
        avg_urlcc_mean_delay.append(np.nanmean(urlcc_mean_delay_vals))
        avg_embb_mean_delay.append(np.nanmean(embb_mean_delay_vals))
        avg_urlcc_mean_wait_n1.append(np.nanmean(urlcc_mean_wait_vals))
        avg_embb_mean_wait_n1.append(np.nanmean(embb_mean_wait_vals))
        avg_urlcc_violation.append(np.nanmean(urlcc_violation_vals))

        rows.append({
            "lambda_urllc": lambda_urllc,
            "lambda_embb": LAMBDA_EMBB_FIXED,
            "urlcc_p95_mean": avg_urlcc_p95[-1],
            "embb_p95_mean": avg_embb_p95[-1],
            "urlcc_loss_mean": avg_urlcc_loss[-1],
            "embb_loss_mean": avg_embb_loss[-1],
            "urlcc_mean_delay": avg_urlcc_mean_delay[-1],
            "embb_mean_delay": avg_embb_mean_delay[-1],
            "urlcc_mean_wait_node1": avg_urlcc_mean_wait_n1[-1],
            "embb_mean_wait_node1": avg_embb_mean_wait_n1[-1],
            "urlcc_violation_prob_1ms": avg_urlcc_violation[-1],
        })

    return {
        "case_name": case_name,
        "lambda_urllc": np.array(URLLC_RATES, dtype=float),
        "urlcc_p95_mean": np.array(avg_urlcc_p95, dtype=float),
        "embb_p95_mean": np.array(avg_embb_p95, dtype=float),
        "urlcc_loss_mean": np.array(avg_urlcc_loss, dtype=float),
        "embb_loss_mean": np.array(avg_embb_loss, dtype=float),
        "urlcc_mean_delay": np.array(avg_urlcc_mean_delay, dtype=float),
        "embb_mean_delay": np.array(avg_embb_mean_delay, dtype=float),
        "urlcc_mean_wait_node1": np.array(avg_urlcc_mean_wait_n1, dtype=float),
        "embb_mean_wait_node1": np.array(avg_embb_mean_wait_n1, dtype=float),
        "urlcc_violation_prob_1ms": np.array(avg_urlcc_violation, dtype=float),
        "rows": rows,
    }


def save_csv(results_dict, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            "lambda_urllc,lambda_embb,"
            "urlcc_p95_mean,embb_p95_mean,"
            "urlcc_loss_mean,embb_loss_mean,"
            "urlcc_mean_delay,embb_mean_delay,"
            "urlcc_mean_wait_node1,embb_mean_wait_node1,"
            "urlcc_violation_prob_1ms\n"
        )
        for row in results_dict["rows"]:
            f.write(
                f"{row['lambda_urllc']},"
                f"{row['lambda_embb']},"
                f"{row['urlcc_p95_mean']},"
                f"{row['embb_p95_mean']},"
                f"{row['urlcc_loss_mean']},"
                f"{row['embb_loss_mean']},"
                f"{row['urlcc_mean_delay']},"
                f"{row['embb_mean_delay']},"
                f"{row['urlcc_mean_wait_node1']},"
                f"{row['embb_mean_wait_node1']},"
                f"{row['urlcc_violation_prob_1ms']}\n"
            )


def plot_results(results_dict, prefix):
    x = results_dict["lambda_urllc"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, 1000.0 * results_dict["urlcc_p95_mean"], marker="o", label="URLLC")
    plt.plot(x, 1000.0 * results_dict["embb_p95_mean"], marker="s", label="eMBB")
    plt.xlabel(r"URLLC arrival rate $\lambda_U$ (packets/s)")
    plt.ylabel("95th-percentile end-to-end delay (ms)")
    plt.title(f"95th-percentile delays vs URLLC rate ({results_dict['case_name']})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_p95_delay.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(x, results_dict["urlcc_loss_mean"], marker="o", label="URLLC")
    plt.plot(x, results_dict["embb_loss_mean"], marker="s", label="eMBB")
    plt.xlabel(r"URLLC arrival rate $\lambda_U$ (packets/s)")
    plt.ylabel("Loss probability")
    plt.title(f"Loss probabilities vs URLLC rate ({results_dict['case_name']})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_loss_prob.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(x, results_dict["urlcc_violation_prob_1ms"], marker="o")
    plt.xlabel(r"URLLC arrival rate $\lambda_U$ (packets/s)")
    plt.ylabel("URLLC delay violation probability")
    plt.title(f"URLLC violation probability vs URLLC rate ({results_dict['case_name']})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_urlcc_violation.png"), dpi=300)
    plt.show()


def print_summary(results_dict):
    print("\n" + "=" * 100)
    print(f"SUMMARY: {results_dict['case_name']}")
    print("=" * 100)
    print(
        f"{'lambda_U':>10} {'p95_URLLC(ms)':>15} {'p95_eMBB(ms)':>15} "
        f"{'loss_URLLC':>15} {'loss_eMBB':>15} {'Pvio_URLLC':>15}"
    )

    for i, lambda_u in enumerate(results_dict["lambda_urllc"]):
        print(
            f"{lambda_u:10.1f} "
            f"{1000.0 * results_dict['urlcc_p95_mean'][i]:15.6f} "
            f"{1000.0 * results_dict['embb_p95_mean'][i]:15.6f} "
            f"{results_dict['urlcc_loss_mean'][i]:15.6e} "
            f"{results_dict['embb_loss_mean'][i]:15.6e} "
            f"{results_dict['urlcc_violation_prob_1ms'][i]:15.6e}"
        )


if __name__ == "__main__":
    results_infinite = run_sweep(
        buffer_size_node1=None,
        case_name="Infinite buffer"
    )
    print_summary(results_infinite)
    save_csv(results_infinite, os.path.join(OUTPUT_DIR, "vary_urllc_infinite.csv"))
    plot_results(results_infinite, "vary_urllc_infinite")

    results_finite = run_sweep(
        buffer_size_node1=BUFFER_SIZE_NODE1,
        case_name=f"Finite buffer (B={BUFFER_SIZE_NODE1})"
    )
    print_summary(results_finite)
    save_csv(results_finite, os.path.join(OUTPUT_DIR, "vary_urllc_finite.csv"))
    plot_results(results_finite, "vary_urllc_finite")