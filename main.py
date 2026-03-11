from simulation import NetworkSimulator
from PARAMETERS import *


def get_param(name, default):
    return globals()[name] if name in globals() else default


def print_results(title, results):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    u = results["URLLC"]
    e = results["eMBB"]

    print("\n[URLLC]")
    print(f"Arrivals                      : {u['arrivals']}")
    print(f"Completed                     : {u['completed']}")
    print(f"Losses                        : {u['losses']}")
    print(f"Mean waiting time at Node 1   : {u['mean_wait_node1']:.6e} s")
    print(f"Mean end-to-end delay         : {u['mean_delay_total']:.6e} s")
    print(f"Mean delay at Node 1          : {u['mean_delay_node1']:.6e} s")
    print(f"Mean delay at Node 2          : {u['mean_delay_node2']:.6e} s")
    print(f"P(delay > 1 ms)               : {u['delay_violation_prob_1ms']:.6e}")

    print("\n[eMBB]")
    print(f"Arrivals                      : {e['arrivals']}")
    print(f"Completed                     : {e['completed']}")
    print(f"Losses                        : {e['losses']}")
    print(f"Mean waiting time at Node 1   : {e['mean_wait_node1']:.6e} s")
    print(f"Mean throughput               : {e['mean_throughput']:.6e} packets/s")
    print(f"Mean end-to-end delay         : {e['mean_delay_total']:.6e} s")
    print(f"Mean delay at Node 1          : {e['mean_delay_node1']:.6e} s")
    print(f"Mean delay at Node 2          : {e['mean_delay_node2']:.6e} s")
    print(f"Loss probability              : {e['loss_prob']:.6e}")


def run_case(buffer_size_node1, case_name):
    sim = NetworkSimulator(
        lambda_urllc=get_param("LAMBDA_URLLC", None),
        lambda_embb=get_param("LAMBDA_EMBB", None),
        mu_r=get_param("MU_R", None),
        mu_b=get_param("MU_B", None),
        sim_time=get_param("T_SIM", None),
        warmup=get_param("WARMUP", 0.0),
        delay_threshold_urllc=get_param("URLLC_DELAY_THRESHOLD", 1e-3),
        buffer_size_node1=buffer_size_node1,
        seed=get_param("SEED", None),
    )

    results = sim.run()
    print_results(case_name, results)
    return results


if __name__ == "__main__":
    run_case(
        buffer_size_node1=None,
        case_name="CASE 1: Node 1 with infinite buffer"
    )

    run_case(
        buffer_size_node1=get_param("BUFFER_SIZE_NODE1", 100),
        case_name="CASE 2: Node 1 with finite buffer"
    )