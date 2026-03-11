import os
import numpy as np
import matplotlib.pyplot as plt

from simulation import NetworkSimulator
from PARAMETERS import *


def get_param(name, default):
    return globals()[name] if name in globals() else default


# ============================================================
# Fixed offered traffic
# ============================================================

LAMBDA_URLLC = get_param("LAMBDA_URLLC", 200.0)
LAMBDA_EMBB = get_param("LAMBDA_EMBB", 3000.0)

SIM_TIME = get_param("T_SIM", 2000.0)
WARMUP = get_param("WARMUP", 200.0)
BUFFER_SIZE_NODE1 = get_param("BUFFER_SIZE_NODE1", 100)
URLLC_DELAY_THRESHOLD = get_param("URLLC_DELAY_THRESHOLD", 1e-3)

NUM_REPETITIONS = 10

OUTPUT_DIR = "results_regimes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Service-rate regimes
# You can change/add pairs here
# ============================================================

REGIMES = [
    {"name": "Balanced",          "mu_r": 5000.0, "mu_b": 10000.0},
    {"name": "Radio-limited",     "mu_r": 3500.0, "mu_b": 12000.0},
    {"name": "Backhaul-limited",  "mu_r": 7000.0, "mu_b": 4000.0},
    {"name": "Fast-both",         "mu_r": 8000.0, "mu_b": 14000.0},
]

# Also sweep total offered load scaling around the baseline arrivals
LOAD_SCALE_VALUES = np.linspace(0.5, 1.4, 10)


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
        "urlcc_p95_delay": p95_urlcc,
        "embb_p95_delay": p95_embb,
        "urlcc_loss_prob": results["URLLC"]["losses"] / results["URLLC"]["arrivals"]
        if results["URLLC"]["arrivals"] > 0 else np.nan,
        "embb_loss_prob": results["eMBB"]["loss_prob"],
        "urlcc_mean_delay": results["URLLC"]["mean_delay_total"],
        "embb_mean_delay": results["eMBB"]["mean_delay_total"],
        "urlcc_violation_prob_1ms": results["URLLC"]["delay_violation_prob_1ms"],
    }


def run_regime(regime, buffer_size_node1):
    rows = []

    for scale in LOAD_SCALE_VALUES:
        lambda_u = scale * LAMBDA_URLLC
        lambda_e = scale * LAMBDA_EMBB

        run_results = []
        for seed in range(NUM_REPETITIONS):
            r = run_one_simulation(
                lambda_urllc=lambda_u,
                lambda_embb=lambda_e,
                mu_r=regime["mu_r"],
                mu_b=regime["mu_b"],
                seed=seed,
                buffer_size_node1=buffer_size_node1,
            )
            run_results.append(r)

        rows.append({
            "scale": scale,
            "lambda_urllc": lambda_u,
            "lambda_embb": lambda_e,
            "urlcc_p95_mean": np.nanmean([r["urlcc_p95_delay"] for r in run_results]),
            "embb_p95_mean": np.nanmean([r["embb_p95_delay"] for r in run_results]),
            "urlcc_loss_mean": np.nanmean([r["urlcc_loss_prob"] for r in run_results]),
            "embb_loss_mean": np.nanmean([r["embb_loss_prob"] for r in run_results]),
            "urlcc_mean_delay": np.nanmean([r["urlcc_mean_delay"] for r in run_results]),
            "embb_mean_delay": np.nanmean([r["embb_mean_delay"] for r in run_results]),
            "urlcc_violation_prob_1ms": np.nanmean([r["urlcc_violation_prob_1ms"] for r in run_results]),
        })

    return rows


def save_regime_csv(all_results, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            "case,scale,lambda_urllc,lambda_embb,mu_r,mu_b,"
            "urlcc_p95_mean,embb_p95_mean,"
            "urlcc_loss_mean,embb_loss_mean,"
            "urlcc_mean_delay,embb_mean_delay,"
            "urlcc_violation_prob_1ms\n"
        )

        for item in all_results:
            regime_name = item["name"]
            mu_r = item["mu_r"]
            mu_b = item["mu_b"]
            for row in item["rows"]:
                f.write(
                    f"{regime_name},"
                    f"{row['scale']},"
                    f"{row['lambda_urllc']},"
                    f"{row['lambda_embb']},"
                    f"{mu_r},"
                    f"{mu_b},"
                    f"{row['urlcc_p95_mean']},"
                    f"{row['embb_p95_mean']},"
                    f"{row['urlcc_loss_mean']},"
                    f"{row['embb_loss_mean']},"
                    f"{row['urlcc_mean_delay']},"
                    f"{row['embb_mean_delay']},"
                    f"{row['urlcc_violation_prob_1ms']}\n"
                )


def plot_metric(all_results, y_key, ylabel, title, filename, multiply_by_1000=False):
    plt.figure(figsize=(8, 5))

    for item in all_results:
        x = np.array([row["scale"] for row in item["rows"]], dtype=float)
        y = np.array([row[y_key] for row in item["rows"]], dtype=float)
        if multiply_by_1000:
            y = 1000.0 * y
        plt.plot(x, y, marker="o", label=item["name"])

    plt.xlabel("Offered-load scaling factor")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    # choose one of these:
    # buffer_case = None
    # case_label = "Infinite buffer"

    buffer_case = BUFFER_SIZE_NODE1
    case_label = f"Finite buffer (B={BUFFER_SIZE_NODE1})"

    all_results = []

    for regime in REGIMES:
        rows = run_regime(regime, buffer_case)
        all_results.append({
            "name": regime["name"],
            "mu_r": regime["mu_r"],
            "mu_b": regime["mu_b"],
            "rows": rows,
        })

    save_regime_csv(all_results, os.path.join(OUTPUT_DIR, "regime_study.csv"))

    plot_metric(
        all_results,
        y_key="urlcc_p95_mean",
        ylabel="URLLC 95th-percentile delay (ms)",
        title=f"URLLC 95th-percentile delay: radio-limited vs backhaul-limited ({case_label})",
        filename=os.path.join(OUTPUT_DIR, "urlcc_p95_regimes.png"),
        multiply_by_1000=True,
    )

    plot_metric(
        all_results,
        y_key="embb_p95_mean",
        ylabel="eMBB 95th-percentile delay (ms)",
        title=f"eMBB 95th-percentile delay: radio-limited vs backhaul-limited ({case_label})",
        filename=os.path.join(OUTPUT_DIR, "embb_p95_regimes.png"),
        multiply_by_1000=True,
    )

    plot_metric(
        all_results,
        y_key="embb_loss_mean",
        ylabel="eMBB loss probability",
        title=f"eMBB loss probability: radio-limited vs backhaul-limited ({case_label})",
        filename=os.path.join(OUTPUT_DIR, "embb_loss_regimes.png"),
        multiply_by_1000=False,
    )

    plot_metric(
        all_results,
        y_key="urlcc_violation_prob_1ms",
        ylabel="URLLC delay violation probability",
        title=f"URLLC delay violation probability: radio-limited vs backhaul-limited ({case_label})",
        filename=os.path.join(OUTPUT_DIR, "urlcc_violation_regimes.png"),
        multiply_by_1000=False,
    )