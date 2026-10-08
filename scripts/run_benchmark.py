"""
CLI entrypoint to execute the VQE Cost Derivative Variance Benchmark:
Hamiltonian-Weighted MUB Ensembles vs. Standard Haar Initialization.
"""

import argparse
import json
from pathlib import Path


from mub.benchmark import (
    run_variance_benchmark,
    plot_variance_benchmark,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run cost derivative variance benchmark for MUB Ensembles vs. Haar Initialization."
    )
    parser.add_argument(
        "-q", "--qubits",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8],
        help="List of qubit counts to benchmark (default: 3 4 5 6 7 8)"
    )
    parser.add_argument(
        "-s", "--samples",
        type=int,
        default=1000,
        help="Number of initialization samples per qubit count (default: 1000)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["all_to_all_dm", "xy_dm", "tfim"],
        default="all_to_all_dm",
        help="Hamiltonian model: 'all_to_all_dm' (All-to-All DM with Kac norm), 'xy_dm' (1D XY with DM), or 'tfim' (default: all_to_all_dm)"
    )
    parser.add_argument(
        "-l", "--layers-factor",
        type=int,
        default=2,
        help="Multiplier for ansatz depth: layers L = layers_factor * n (default: 2, i.e., L=2n)"
    )
    parser.add_argument(
        "-p", "--param-idx",
        type=str,
        default="mid",
        help="Ansatz parameter index k or 'mid'/'middle' (default: mid)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="outputs/gradient_variance_benchmark.pdf",
        help="Path for saving benchmark plot (default: outputs/gradient_variance_benchmark.pdf)"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="outputs/gradient_variance_benchmark.json",
        help="Path for saving numerical benchmark data (default: outputs/gradient_variance_benchmark.json)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["mean_param_var", "grad_norm_sq", "single_param"],
        default="mean_param_var",
        help="Benchmark metric: 'mean_param_var' (Mean parameter variance (1/P) sum Var[∂_k C]), 'grad_norm_sq' (variance of ||∇C||^2), or 'single_param' (variance of ∂_k C) (default: mean_param_var)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--diff-method",
        type=str,
        default="adjoint",
        choices=["adjoint", "backprop"],
        help="PennyLane differentiation method (default: adjoint)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PennyLane simulator device (default: auto, detects lightning.qubit)"
    )

    args = parser.parse_args()

    param_val = int(args.param_idx) if args.param_idx.isdigit() else args.param_idx

    import time
    total_start_time = time.time()

    results = run_variance_benchmark(
        qubits_list=args.qubits,
        num_samples=args.samples,
        hamiltonian_type=args.model,
        layers_factor=args.layers_factor,
        metric=args.metric,
        diff_method=args.diff_method,
        device_name=args.device,
        param_idx=param_val,
        seed=args.seed,
        verbose=True
    )

    total_elapsed_time = time.time() - total_start_time

    # Save JSON
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Numerical results saved to: {json_path}")

    # Save plot
    out_img = plot_variance_benchmark(results, output_path=args.output)
    print(f"[OK] Benchmark plot saved to: {out_img}")

    # Print summary table
    metric_type = results.get("metric", "")
    if metric_type == "mean_param_var":
        metric_label = "(1/P) ∑ Var[∂_k C]"
    elif metric_type == "grad_norm_sq":
        metric_label = "Var[||∇C||^2]"
    else:
        metric_label = "Var[∂_k C]"

    print("\n" + "=" * 105)
    print(f" SUMMARY TABLE: GRADIENT METRIC VARIANCE {metric_label}")
    print("=" * 105)
    print(f"{'n':<4} | {'Param / Mode':<12} | {'Active Sets':<12} | {'Haar Var ± SE':<24} | {'MUB Var ± SE':<24} | {'Ratio (MUB/Haar)':<16}")
    print("-" * 105)
    haar_errs = results.get("haar_var_err", [0.0] * len(results["qubits"]))
    mub_errs = results.get("mub_var_err", [0.0] * len(results["qubits"]))
    for n, h_var, h_err, m_var, m_err in zip(results["qubits"], results["haar_var"], haar_errs, results["mub_var"], mub_errs):
        if metric_type == "mean_param_var":
            mode_label = "Mean(P)"
        elif metric_type == "grad_norm_sq":
            mode_label = "||∇C||^2"
        else:
            mode_label = f"k={results['evaluated_param_indices'].get(n, '-')}"
        act = results["active_stabilizer_counts"].get(n, "-")
        ratio = m_var / h_var if h_var > 0 else float("inf")
        h_str = f"{h_var:.4e} ± {h_err:.1e}"
        m_str = f"{m_var:.4e} ± {m_err:.1e}"
        print(f"{n:<4} | {mode_label:<12} | {act:<12} | {h_str:<24} | {m_str:<24} | {ratio:<16.3f}")
    print("=" * 105)
    print(f"\n[BENCHMARK FINISHED] Total time elapsed: {total_elapsed_time:.2f} seconds ({total_elapsed_time / 60:.2f} minutes).")


if __name__ == "__main__":
    main()
