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
        type=int,
        default=0,
        help="Ansatz parameter index k to evaluate ∂_k C for (default: 0)"
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
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    results = run_variance_benchmark(
        qubits_list=args.qubits,
        num_samples=args.samples,
        hamiltonian_type=args.model,
        layers_factor=args.layers_factor,
        param_idx=args.param_idx,
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
    print("\n" + "=" * 80)
    print(" SUMMARY TABLE: COST DERIVATIVE VARIANCE Var[∂_k C]")
    print("=" * 80)
    print(f"{'n':<4} | {'Active Sets':<12} | {'Haar Var':<16} | {'MUB Var':<16} | {'Ratio (MUB/Haar)':<16}")
    print("-" * 80)
    for n, h_var, m_var in zip(results["qubits"], results["haar_var"], results["mub_var"]):
        act = results["active_stabilizer_counts"].get(n, "-")
        ratio = m_var / h_var if h_var > 0 else float("inf")
        print(f"{n:<4} | {act:<12} | {h_var:<16.6e} | {m_var:<16.6e} | {ratio:<16.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
