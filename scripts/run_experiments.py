"""Run named benchmark experiments defined in src/experiments.py.

Usage:
    python scripts/run_experiments.py              # run all experiments
    python scripts/run_experiments.py tfim         # run only the named ones
    python scripts/run_experiments.py --list       # list available experiments
    python scripts/run_experiments.py --refit ising_theta1   # re-fit/re-plot from saved JSON (no re-run)
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.experiments import EXPERIMENTS  # noqa: E402
from src.benchmark import (  # noqa: E402
    add_decay_fits,
    plot_variance_benchmark,
    run_variance_benchmark,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIT_N_MIN = 7


def run_experiment(name: str) -> None:
    params = EXPERIMENTS[name]
    print(f"\n##### Experiment: {name} #####")
    results = run_variance_benchmark(verbose=True, **params)
    results["experiment"] = name
    results["seed"] = params["seed"]
    save_experiment(name, results)


def save_experiment(name: str, results: dict) -> None:
    """Saves JSON and plot. Single-parameter experiments also get an A*n*b^(-n) fit for n >= 7."""
    if results.get("metric") == "single_param":
        add_decay_fits(results, n_min=FIT_N_MIN)
        print_fits(name, results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{name}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_path = plot_variance_benchmark(results, output_path=OUTPUT_DIR / f"{name}.pdf")
    print(f"[OK] {name}: {json_path} and {plot_path}")


def print_fits(name: str, results: dict) -> None:
    print(f"\nFit Var ~ A * n * b^(-n), n >= {FIT_N_MIN}  [{name}]")
    for key, fit in results["fits"].items():
        if fit is None:
            print(f"  {key:<5}: not fitted (non-positive variance or too few points)")
        else:
            print(f"  {key:<5}: A = {fit['A']:.4g} ± {fit['A_err']:.2g}, b = {fit['b']:.4f} ± {fit['b_err']:.1g}")


def refit_experiment(name: str) -> None:
    """Re-applies the fit and re-plots from the saved JSON, without re-running the benchmark."""
    json_path = OUTPUT_DIR / f"{name}.json"
    with json_path.open(encoding="utf-8") as f:
        results = json.load(f)
    save_experiment(name, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run named benchmark experiments.")
    parser.add_argument("names", nargs="*", help="Experiments to run (default: all)")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    parser.add_argument("--refit", action="store_true",
                        help="Re-fit and re-plot from saved JSON instead of re-running the benchmark")
    args = parser.parse_args()

    if args.list:
        for name, params in EXPERIMENTS.items():
            print(f"{name}: {params['hamiltonian_type']}, metric={params['metric']}, n={params['qubits_list'][0]}-{params['qubits_list'][-1]}, "
                  f"samples={params['num_samples']}")
        return

    unknown = [n for n in args.names if n not in EXPERIMENTS]
    if unknown:
        parser.error(f"Unknown experiment(s): {', '.join(unknown)}. Available: {', '.join(EXPERIMENTS)}")

    for name in args.names or list(EXPERIMENTS):
        if args.refit:
            refit_experiment(name)
        else:
            run_experiment(name)


if __name__ == "__main__":
    main()
