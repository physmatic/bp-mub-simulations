"""Run named benchmark experiments defined in src/experiments.py.

Usage:
    python scripts/run_experiments.py                   # run all current experiments
    python scripts/run_experiments.py tfim_ryrz         # run only the named ones
    python scripts/run_experiments.py --list            # list available experiments
    python scripts/run_experiments.py --refit NAME      # re-fit/re-plot from saved JSON (no re-run)

Staged runs (small n first, then extend):
    python scripts/run_experiments.py --qubits 2-12 --jobs 4 --notify
    python scripts/run_experiments.py --qubits 13-16 --merge --jobs 4 --notify

    --qubits overrides the experiment's qubit range; --merge combines the new qubit counts
    with outputs/<name>.json (configs must match). Each n is seeded independently, so the
    merged result equals a single full run.

Legacy (Ry+CZ, L=2n) experiments live in LEGACY_EXPERIMENTS and write to outputs/old/:
    python scripts/run_experiments.py --legacy --list
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.experiments import EXPERIMENTS, LEGACY_EXPERIMENTS  # noqa: E402
from src.benchmark import (  # noqa: E402
    add_decay_fits,
    plot_variance_benchmark,
    run_variance_benchmark,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LEGACY_OUTPUT_DIR = OUTPUT_DIR / "old"
FIT_N_MIN = 7

# Fields that must agree for two partial runs to be merged.
MERGE_CONFIG_KEYS = ("hamiltonian_name", "metric", "layers_factor", "gate_set", "num_samples",
                     "seed", "param_idx_spec", "diff_method")
PER_N_LISTS = ("haar_var", "haar_var_err", "haar_mean", "mub_var", "mub_var_err", "mub_mean")
PER_N_DICTS = ("details", "active_stabilizer_counts", "evaluated_param_indices")


def parse_qubits(spec: str) -> list:
    """'2-12' -> [2..12]; '2,4,8' -> [2, 4, 8]; '13-16,18' also works."""
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def merge_results(old: dict, new: dict) -> dict:
    """Combine two result dicts over disjoint (or overlapping: new wins) qubit counts."""
    mismatched = [k for k in MERGE_CONFIG_KEYS if old.get(k) != new.get(k)]
    if mismatched:
        raise ValueError("Cannot merge, config differs in: "
                         + ", ".join(f"{k} (old={old.get(k)!r}, new={new.get(k)!r})" for k in mismatched))
    rows = {}
    for res in (old, new):
        for i, n in enumerate(res["qubits"]):
            rows[int(n)] = {
                **{k: res[k][i] for k in PER_N_LISTS},
                **{k: res.get(k, {}).get(str(n)) for k in PER_N_DICTS},
            }
    merged = dict(new)
    merged["qubits"] = sorted(rows)
    for k in PER_N_LISTS:
        merged[k] = [rows[n][k] for n in merged["qubits"]]
    for k in PER_N_DICTS:
        merged[k] = {str(n): rows[n][k] for n in merged["qubits"] if rows[n][k] is not None}
    return merged


def notify(title: str, message: str, open_path: Path = None) -> None:
    """macOS notification; the figure opens in Preview in the background (no focus stealing)."""
    if sys.platform != "darwin":
        return
    script = f'display notification {json.dumps(message)} with title {json.dumps(title)} sound name "Glass"'
    subprocess.run(["osascript", "-e", script], check=False)
    if open_path is not None:
        subprocess.run(["open", "-g", "-a", "Preview", str(open_path)], check=False)


def run_experiment(name: str, registry: dict, out_dir: Path, qubits=None, merge=False, do_notify=False) -> Path:
    params = dict(registry[name])
    if qubits is not None:
        params["qubits_list"] = qubits
    print(f"\n##### Experiment: {name} (n = {params['qubits_list']}) #####", flush=True)
    results = run_variance_benchmark(verbose=True, **params)
    results["experiment"] = name
    # JSON round-trip so int dict keys become strings, matching what is loaded from disk.
    results = json.loads(json.dumps(results))

    json_path = out_dir / f"{name}.json"
    if merge and json_path.exists():
        with json_path.open(encoding="utf-8") as f:
            results = merge_results(json.load(f), results)
    plot_path = save_experiment(name, results, out_dir)
    if do_notify:
        notify(f"Figure ready: {name}", f"n = {results['qubits'][0]}..{results['qubits'][-1]}", plot_path)
    return plot_path


def save_experiment(name: str, results: dict, out_dir: Path) -> Path:
    """Saves JSON and plot. Single-parameter experiments also get an A*n*b^(-n) fit for n >= 7."""
    if results.get("metric") == "single_param":
        add_decay_fits(results, n_min=FIT_N_MIN)
        print_fits(name, results)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_path = plot_variance_benchmark(results, output_path=out_dir / f"{name}.pdf")
    print(f"[OK] {name}: {json_path} and {plot_path}", flush=True)
    return plot_path


def print_fits(name: str, results: dict) -> None:
    print(f"\nFit Var ~ A * n * b^(-n), n >= {FIT_N_MIN}  [{name}]")
    for key, fit in results["fits"].items():
        if fit is None:
            print(f"  {key:<5}: not fitted (non-positive variance or too few points)")
        else:
            print(f"  {key:<5}: A = {fit['A']:.4g} ± {fit['A_err']:.2g}, b = {fit['b']:.4f} ± {fit['b_err']:.1g}")


def refit_experiment(name: str, out_dir: Path) -> None:
    """Re-applies the fit and re-plots from the saved JSON, without re-running the benchmark."""
    with (out_dir / f"{name}.json").open(encoding="utf-8") as f:
        results = json.load(f)
    save_experiment(name, results, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run named benchmark experiments.")
    parser.add_argument("names", nargs="*", help="Experiments to run (default: all)")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    parser.add_argument("--refit", action="store_true",
                        help="Re-fit and re-plot from saved JSON instead of re-running the benchmark")
    parser.add_argument("--legacy", action="store_true",
                        help="Use LEGACY_EXPERIMENTS (Ry+CZ, L=2n) and outputs/old/")
    parser.add_argument("--qubits", type=parse_qubits, default=None,
                        help="Override qubit range, e.g. '2-12' or '13-16' or '2,4,8'")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing outputs/<name>.json instead of overwriting")
    parser.add_argument("--jobs", type=int, default=1, help="Experiments to run in parallel (default 1)")
    parser.add_argument("--notify", action="store_true",
                        help="macOS notification + open the PDF in Preview when each figure is ready")
    args = parser.parse_args()

    registry = LEGACY_EXPERIMENTS if args.legacy else EXPERIMENTS
    out_dir = LEGACY_OUTPUT_DIR if args.legacy else OUTPUT_DIR

    if args.list:
        for name, params in registry.items():
            print(f"{name}: {params['hamiltonian_type']}, metric={params['metric']}, gates={params['gate_set']}, "
                  f"L={params['layers_factor']}n, n={params['qubits_list'][0]}-{params['qubits_list'][-1]}, "
                  f"samples={params['num_samples']}")
        return

    unknown = [n for n in args.names if n not in registry]
    if unknown:
        parser.error(f"Unknown experiment(s): {', '.join(unknown)}. Available: {', '.join(registry)}")
    names = args.names or list(registry)

    if args.refit:
        for name in names:
            refit_experiment(name, out_dir)
        return

    if args.jobs <= 1:
        for name in names:
            run_experiment(name, registry, out_dir, args.qubits, args.merge, args.notify)
        return

    # One process per experiment; keep each simulator single-threaded to avoid oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_experiment, name, registry, out_dir, args.qubits, args.merge, args.notify): name
                   for name in names}
        for fut in as_completed(futures):
            fut.result()  # re-raise worker errors
            print(f"[DONE] {futures[fut]}", flush=True)
    if args.notify:
        notify("All experiments finished", ", ".join(names))


if __name__ == "__main__":
    main()
