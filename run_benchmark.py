"""
CLI entrypoint to execute the VQE Cost Derivative Variance Benchmark:
Hamiltonian-Weighted MUB Ensembles vs. Standard Haar Initialization.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project modules are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "MUB_Circuits_generator"))

from src.MUB_Circuits_generator.benchmark import (
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
        default=[2, 3, 4, 5, 6],
        help="List of qubit counts to benchmark (default: 2 3 4 5 6)"
    )
    parser.add_argument(
        "-s", "--samples",
        type=int,
        default=200,
        help="Number of initialization samples per qubit count (default: 200)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["xy_dm", "tfim"],
        default="xy_dm",
        help="Hamiltonian model: 'xy_dm' (1D XY with DM interaction) or 'tfim' (default: xy_dm)"
    )
    parser.add_argument(
        "-l", "--layers-factor",
        type=int,
        default=2,
        help="Multiplier for ansatz depth: layers L = layers_factor * n (default: 2)"
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
        default="outputs/gradient_variance_benchmark.png",
        help="Path for saving benchmark plot (default: outputs/gradient_variance_benchmark.png)"
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
        choices=["grad_norm_sq", "single_param"],
        default="grad_norm_sq",
        help="Benchmark metric: 'grad_norm_sq' (variance of ||∇C||^2) or 'single_param' (variance of ∂_k C) (default: grad_norm_sq)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    param_val = int(args.param_idx) if args.param_idx.isdigit() else args.param_idx

    results = run_variance_benchmark(
        qubits_list=args.qubits,
        num_samples=args.samples,
        hamiltonian_type=args.model,
        layers_factor=args.layers_factor,
        metric=args.metric,
        param_idx=param_val,
        seed=args.seed,
        verbose=True
    )

    # Save plot
    out_img = plot_variance_benchmark(results, output_path=args.output)
    print(f"\n[OK] Benchmark plot saved to: {out_img}")

    # Save JSON
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Numerical results saved to: {json_path}")

    # Print summary table
    is_norm_sq = results.get("metric", "") == "grad_norm_sq"
    metric_label = "Var[||∇C||^2]" if is_norm_sq else "Var[∂_k C]"
    print("\n" + "=" * 90)
    print(f" SUMMARY TABLE: GRADIENT METRIC VARIANCE {metric_label}")
    print("=" * 90)
    print(f"{'n':<4} | {'Param / Mode':<12} | {'Active Sets':<12} | {'Haar Var':<16} | {'MUB Var':<16} | {'Ratio (MUB/Haar)':<16}")
    print("-" * 90)
    for n, h_var, m_var in zip(results["qubits"], results["haar_var"], results["mub_var"]):
        mode_label = "||∇C||^2" if is_norm_sq else f"k={results['evaluated_param_indices'].get(n, '-')}"
        act = results["active_stabilizer_counts"].get(n, "-")
        ratio = m_var / h_var if h_var > 0 else float("inf")
        print(f"{n:<4} | {mode_label:<12} | {act:<12} | {h_var:<16.6e} | {m_var:<16.6e} | {ratio:<16.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
