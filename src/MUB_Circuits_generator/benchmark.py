"""
Cost Derivative Variance Benchmark.

Compares the variance of cost function derivatives Var[∂_k C] between:
1. Standard Haar random initialization (uniform angles θ ~ [0, 2π)^P starting in |0...0>)
2. Hamiltonian-Weighted MUB Ensemble initialization (starting in |psi_k^j> = U(j)|k> with
   (j, k) sampled according to w_j and Uniform(0, 2^n - 1), evaluated at θ = 0).

Supports single-parameter shift evaluation:
    ∂_k C = [ C(θ + (π/2) e_k) - C(θ - (π/2) e_k) ] / 2
as well as full parameter vector evaluation.
"""

from typing import Callable, Dict, List, Optional, Sequence, Union
from pathlib import Path
import time
import numpy as np
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .ansatz import hardware_efficient_ansatz, get_num_hea_params
    from .hamiltonians import build_tfim_hamiltonian, build_xy_dm_hamiltonian, build_all_to_all_dm_hamiltonian
    from .mub_weights import compute_mub_weights, count_active_stabilizer_sets
    from .pauli_to_j import p_to_j
    from .state_preparation import sample_mub_basis_and_state, prepare_mub_state
except ImportError:
    from ansatz import hardware_efficient_ansatz, get_num_hea_params
    from hamiltonians import build_tfim_hamiltonian, build_xy_dm_hamiltonian, build_all_to_all_dm_hamiltonian
    from mub_weights import compute_mub_weights, count_active_stabilizer_sets
    from pauli_to_j import p_to_j
    from state_preparation import sample_mub_basis_and_state, prepare_mub_state


def evaluate_single_param_shift(
    circuit_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    param_idx: int = 0
) -> float:
    """
    Computes ∂_{param_idx} C using the exact two-point parameter-shift rule:
        ∂_k C = (C(params + (π/2) e_k) - C(params - (π/2) e_k)) / 2.0
    """
    shift = np.zeros_like(params)
    shift[param_idx] = np.pi / 2.0
    val_plus = circuit_fn(params + shift)
    val_minus = circuit_fn(params - shift)
    return float((val_plus - val_minus) / 2.0)


def compute_bootstrap_variance_error(
    data: np.ndarray,
    is_vector: bool = True,
    num_resamples: int = 100,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Computes the bootstrap standard error of the sample variance across trials.

    Parameters
    ----------
    data : np.ndarray
        Array of sample measurements. Shape (N, P) if is_vector else (N,).
    is_vector : bool, default=True
        If True, metric is the mean parameter variance (1/P) sum Var[∂_k C].
        If False, metric is the variance of the 1D sample array.
    num_resamples : int, default=100
        Number of bootstrap resamples.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    float
        Standard error of the variance metric.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(data)
    if N < 2:
        return 0.0

    boot_indices = rng.integers(0, N, size=(num_resamples, N))
    boot_stats = []
    for b in range(num_resamples):
        sample = data[boot_indices[b]]
        if is_vector:
            stat = float(np.mean(np.var(sample, axis=0, ddof=1)))
        else:
            stat = float(np.var(sample, ddof=1))
        boot_stats.append(stat)

    return float(np.std(boot_stats, ddof=1))


def run_variance_benchmark(
    qubits_list: Sequence[int] = (3, 4, 5, 6, 7, 8),
    num_samples: int = 1000,
    hamiltonian_type: Union[str, Callable[[int], qml.Hamiltonian]] = "all_to_all_dm",
    layers_factor: int = 2,
    layers_fn: Optional[Callable[[int], int]] = None,
    metric: str = "mean_param_var",
    diff_method: str = "adjoint",
    device_name: str = "auto",
    param_idx: Union[int, str, Callable[[int, int], int]] = "mid",
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Union[List[int], List[float], Dict]]:
    """
    Sweeps over qubit counts and computes empirical variance of cost function
    gradient metrics for both Haar initialization and Hamiltonian-Weighted
    MUB Ensemble initialization.

    Parameters
    ----------
    qubits_list : Sequence[int]
        List of qubit counts to benchmark, e.g. [3, 4, 5, 6, 7, 8].
    num_samples : int
        Number of random initialization trials per qubit count.
    hamiltonian_type : str or Callable, default="all_to_all_dm"
        Hamiltonian to benchmark: "all_to_all_dm", "xy_dm", or "tfim",
        or a custom callable n -> qml.Hamiltonian.
    layers_factor : int, default=2
        Multiplier for ansatz depth: layers L = layers_factor * n.
    layers_fn : Callable[[int], int], optional
        Custom function n -> layers. Overrides layers_factor if provided.
    metric : str, default="mean_param_var"
        Metric to evaluate:
        - "mean_param_var" / "mean_var": Mean parameter variance (1/P) sum_{k} Var[∂_k C].
        - "grad_norm_sq" / "norm_sq": Variance of squared gradient norm Var[||∇C||^2].
        - "single_param": Variance of partial derivative Var[∂_k C] at param_idx.
    diff_method : str, default="adjoint"
        PennyLane differentiation method: "adjoint" (fastest, O(1) memory gate-by-gate backwards pass)
        or "backprop" (standard reverse-mode automatic differentiation).
    device_name : str, default="auto"
        PennyLane device: "auto" (detects "lightning.qubit" if available, else "default.qubit"),
        "lightning.qubit", or "default.qubit".
    param_idx : int, str, or Callable, default="mid"
        Parameter index k if metric="single_param".
        Can be an int, 'mid' / 'middle' for (L // 2) * n + (n // 2),
        or a callable (n, L) -> k.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress during benchmarking.

    Returns
    -------
    dict
        Benchmark results containing:
        - "qubits": list of n values
        - "hamiltonian_name": name of Hamiltonian model
        - "layers_factor": depth factor
        - "metric": metric evaluated
        - "diff_method": differentiation method used
        - "device_name": backend device used
        - "haar_var": list of Var_Haar[metric]
        - "haar_mean": list of Mean_Haar[metric]
        - "mub_var": list of Var_MUB[metric]
        - "mub_mean": list of Mean_MUB[metric]
        - "details": dict mapping n to detailed sample lists and metadata
    """
    if layers_fn is None:
        layers_fn = lambda n: layers_factor * n

    if isinstance(hamiltonian_type, str):
        if hamiltonian_type.lower() in ["all_to_all_dm", "all_to_all", "dm_all"]:
            ham_builder = lambda n, wires: build_all_to_all_dm_hamiltonian(n, Jx=1.0, Jy=0.5, D=0.8, h=1.0, wires=wires)
            ham_name = "All-to-All DM Model (Jx=1.0, Jy=0.5, D=0.8, h=1.0, 1/n)"
        elif hamiltonian_type.lower() in ["xy_dm", "xy", "dm"]:
            ham_builder = lambda n, wires: build_xy_dm_hamiltonian(n, Jx=1.0, Jy=0.5, D=0.8, h=1.0, wires=wires)
            ham_name = "1D XY-DM Model ($J_x=1.0, J_y=0.5, D=0.8, h=1.0$)"
        elif hamiltonian_type.lower() == "tfim":
            ham_builder = lambda n, wires: build_tfim_hamiltonian(n, J=1.0, h=1.0, wires=wires)
            ham_name = "1D TFIM Model ($J=1.0, h=1.0$)"
        else:
            raise ValueError(f"Unknown hamiltonian_type '{hamiltonian_type}'. Choose 'all_to_all_dm', 'xy_dm', or 'tfim'.")
    else:
        ham_builder = hamiltonian_type
        ham_name = getattr(hamiltonian_type, "__name__", "Custom Hamiltonian")

    is_mean_var = metric.lower() in ["mean_param_var", "mean_var", "mean", "grad_mean_var", "default"]
    is_norm_sq = metric.lower() in ["grad_norm_sq", "norm_sq", "norm2", "grad_norm2"]
    rng = np.random.default_rng(seed)

    actual_metric = "mean_param_var" if is_mean_var else ("grad_norm_sq" if is_norm_sq else "single_param")

    # Resolve target device
    if device_name.lower() == "auto":
        try:
            import pennylane_lightning
            resolved_dev = "lightning.qubit"
        except ImportError:
            resolved_dev = "default.qubit"
    else:
        resolved_dev = device_name

    results = {
        "qubits": list(qubits_list),
        "num_samples": num_samples,
        "metric": actual_metric,
        "diff_method": diff_method,
        "device_name": resolved_dev,
        "param_idx_spec": str(param_idx) if not callable(param_idx) else "callable",
        "evaluated_param_indices": {},
        "hamiltonian_name": ham_name,
        "layers_factor": layers_factor,
        "haar_var": [],
        "haar_var_err": [],
        "haar_mean": [],
        "mub_var": [],
        "mub_var_err": [],
        "mub_mean": [],
        "active_stabilizer_counts": {},
        "details": {}
    }

    if is_mean_var:
        metric_name = "(1/P) sum Var[∂_k C]"
        col_header = "Haar Var ± SE"
        col_header_mub = "MUB Var ± SE"
    elif is_norm_sq:
        metric_name = "Var[||∇C||^2]"
        col_header = "Haar Var ± SE"
        col_header_mub = "MUB Var ± SE"
    else:
        metric_name = f"Var[∂_k C] ({param_idx})"
        col_header = "Haar Var ± SE"
        col_header_mub = "MUB Var ± SE"

    if verbose:
        print("=" * 95)
        print(f" COST GRADIENT VARIANCE BENCHMARK: MUB ENSEMBLE vs. HAAR INITIALIZATION")
        print(f" Hamiltonian: {ham_name}")
        print(f" Device: {resolved_dev} | Diff Method: {diff_method} | Depth: L = {layers_factor}n | Samples: {num_samples}")
        print("=" * 95)
        print(f"{'n':<4} | {'L':<4} | {'Params':<6} | {'Active':<7} | {col_header:<22} | {col_header_mub:<22} | {'Time (s)':<8}")
        print("-" * 95)

    for n in qubits_list:
        t0 = time.time()
        layers = layers_fn(n)
        num_p = get_num_hea_params(n, layers)
        wires = list(range(n))

        H = ham_builder(n, wires)
        weights = compute_mub_weights(H, p_to_j, wires=wires)
        active_count = len(weights)
        results["active_stabilizer_counts"][n] = active_count
        dev = qml.device(resolved_dev, wires=wires)

        if is_mean_var:
            # 1. Haar Initialization: evaluate full gradient vectors
            @qml.qnode(dev, diff_method=diff_method)
            def haar_circuit(p):
                hardware_efficient_ansatz(p, wires=wires, layers=layers)
                return qml.expval(H)

            grad_haar_fn = qml.grad(haar_circuit)
            haar_grads = []
            for _ in range(num_samples):
                theta = qml.numpy.array(rng.uniform(0.0, 2.0 * np.pi, size=num_p), requires_grad=True)
                g = grad_haar_fn(theta)
                haar_grads.append(np.array(g, dtype=float))

            haar_grads = np.array(haar_grads)  # shape (num_samples, num_p)
            haar_param_vars = np.var(haar_grads, axis=0, ddof=1)  # shape (num_p,)
            h_var = float(np.mean(haar_param_vars))
            h_var_err = compute_bootstrap_variance_error(haar_grads, is_vector=True, rng=rng)
            h_mean = float(np.mean(haar_grads))

            # 2. MUB Initialization: evaluate full gradient vectors
            p0 = qml.numpy.zeros(num_p, requires_grad=True)
            mub_grads = []
            for _ in range(num_samples):
                j, k = sample_mub_basis_and_state(weights, n, rng=rng)

                @qml.qnode(dev, diff_method=diff_method)
                def mub_circuit(p, _j=j, _k=k):
                    prepare_mub_state(n, _j, _k, wires=wires)
                    hardware_efficient_ansatz(p, wires=wires, layers=layers)
                    return qml.expval(H)

                g = qml.grad(mub_circuit)(p0)
                mub_grads.append(np.array(g, dtype=float))

            mub_grads = np.array(mub_grads)  # shape (num_samples, num_p)
            mub_param_vars = np.var(mub_grads, axis=0, ddof=1)  # shape (num_p,)
            m_var = float(np.mean(mub_param_vars))
            m_var_err = compute_bootstrap_variance_error(mub_grads, is_vector=True, rng=rng)
            m_mean = float(np.mean(mub_grads))

            details_entry = {
                "layers": layers,
                "num_params": num_p,
                "haar_var_err": h_var_err,
                "mub_var_err": m_var_err,
                "haar_param_vars": haar_param_vars.tolist(),
                "mub_param_vars": mub_param_vars.tolist(),
            }

        elif is_norm_sq:
            # 1. Haar Initialization - Full gradient norm squared
            @qml.qnode(dev, diff_method=diff_method)
            def haar_circuit(p):
                hardware_efficient_ansatz(p, wires=wires, layers=layers)
                return qml.expval(H)

            grad_haar_fn = qml.grad(haar_circuit)
            haar_vals = []
            for _ in range(num_samples):
                theta = qml.numpy.array(rng.uniform(0.0, 2.0 * np.pi, size=num_p), requires_grad=True)
                g = grad_haar_fn(theta)
                haar_vals.append(float(np.sum(g**2)))

            haar_vals = np.array(haar_vals)
            h_var = float(np.var(haar_vals, ddof=1))
            h_var_err = compute_bootstrap_variance_error(haar_vals, is_vector=False, rng=rng)
            h_mean = float(np.mean(haar_vals))

            # 2. MUB Initialization - Full gradient norm squared
            p0 = qml.numpy.zeros(num_p, requires_grad=True)
            mub_vals = []
            for _ in range(num_samples):
                j, k = sample_mub_basis_and_state(weights, n, rng=rng)

                @qml.qnode(dev, diff_method=diff_method)
                def mub_circuit(p, _j=j, _k=k):
                    prepare_mub_state(n, _j, _k, wires=wires)
                    hardware_efficient_ansatz(p, wires=wires, layers=layers)
                    return qml.expval(H)

                g = qml.grad(mub_circuit)(p0)
                mub_vals.append(float(np.sum(g**2)))

            mub_vals = np.array(mub_vals)
            m_var = float(np.var(mub_vals, ddof=1))
            m_var_err = compute_bootstrap_variance_error(mub_vals, is_vector=False, rng=rng)
            m_mean = float(np.mean(mub_vals))

            details_entry = {
                "layers": layers,
                "num_params": num_p,
                "haar_var_err": h_var_err,
                "mub_var_err": m_var_err,
                "haar_vals": haar_vals.tolist(),
                "mub_vals": mub_vals.tolist(),
            }

        else:
            # Single parameter shift
            if isinstance(param_idx, str) and param_idx.lower() in ["mid", "middle"]:
                mid_layer = layers // 2
                mid_wire = n // 2
                actual_k = mid_layer * n + mid_wire
            elif callable(param_idx):
                actual_k = int(param_idx(n, layers))
            else:
                actual_k = int(param_idx)

            if actual_k >= num_p or actual_k < 0:
                raise ValueError(f"param_idx={actual_k} out of range [0, {num_p}) for n={n}, layers={layers}.")
            results["evaluated_param_indices"][n] = actual_k

            @qml.qnode(dev)
            def haar_circuit(p):
                hardware_efficient_ansatz(p, wires=wires, layers=layers)
                return qml.expval(H)

            haar_vals = []
            for _ in range(num_samples):
                theta = rng.uniform(0.0, 2.0 * np.pi, size=num_p)
                grad_val = evaluate_single_param_shift(haar_circuit, theta, param_idx=actual_k)
                haar_vals.append(grad_val)

            haar_vals = np.array(haar_vals)
            h_var = float(np.var(haar_vals, ddof=1))
            h_var_err = compute_bootstrap_variance_error(haar_vals, is_vector=False, rng=rng)
            h_mean = float(np.mean(haar_vals))

            p0 = np.zeros(num_p, dtype=float)
            mub_vals = []
            for _ in range(num_samples):
                j, k = sample_mub_basis_and_state(weights, n, rng=rng)

                @qml.qnode(dev)
                def mub_circuit(p, _j=j, _k=k):
                    prepare_mub_state(n, _j, _k, wires=wires)
                    hardware_efficient_ansatz(p, wires=wires, layers=layers)
                    return qml.expval(H)

                grad_val = evaluate_single_param_shift(mub_circuit, p0, param_idx=actual_k)
                mub_vals.append(grad_val)

            mub_vals = np.array(mub_vals)
            m_var = float(np.var(mub_vals, ddof=1))
            m_var_err = compute_bootstrap_variance_error(mub_vals, is_vector=False, rng=rng)
            m_mean = float(np.mean(mub_vals))

            details_entry = {
                "layers": layers,
                "num_params": num_p,
                "param_k": actual_k,
                "haar_var_err": h_var_err,
                "mub_var_err": m_var_err,
                "haar_vals": haar_vals.tolist(),
                "mub_vals": mub_vals.tolist(),
            }

        dt = time.time() - t0

        results["haar_var"].append(h_var)
        results["haar_var_err"].append(h_var_err)
        results["haar_mean"].append(h_mean)
        results["mub_var"].append(m_var)
        results["mub_var_err"].append(m_var_err)
        results["mub_mean"].append(m_mean)
        details_entry["runtime_sec"] = dt
        results["details"][n] = details_entry

        if verbose:
            h_str = f"{h_var:.4e} ± {h_var_err:.1e}"
            m_str = f"{m_var:.4e} ± {m_var_err:.1e}"
            print(f"{n:<4} | {layers:<4} | {num_p:<6} | {active_count:<7} | {h_str:<22} | {m_str:<22} | {dt:<8.2f}")

    if verbose:
        print("=" * 85)

    return results


def plot_variance_benchmark(
    results: Dict,
    output_path: Union[str, Path] = "outputs/gradient_variance_benchmark.pdf",
    show: bool = False
) -> Path:
    """
    Generates a semi-log plot (log10(Var) vs n) of cost derivative variances
    comparing Haar random initialization with Hamiltonian-Weighted MUB Ensembles.

    Parameters
    ----------
    results : dict
        Output from run_variance_benchmark.
    output_path : str or Path
        Destination path for saving the figure.
    show : bool
        Whether to display the plot interactively.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qubits = np.array(results["qubits"])
    haar_var = np.array(results["haar_var"])
    mub_var = np.array(results["mub_var"])
    haar_err = np.array(results["haar_var_err"]) if "haar_var_err" in results and len(results["haar_var_err"]) == len(haar_var) else None
    mub_err = np.array(results["mub_var_err"]) if "mub_var_err" in results and len(results["mub_var_err"]) == len(mub_var) else None

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Plot MUB curve and 95% confidence band
    ax.plot(
        qubits,
        mub_var,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color="#1E88E5",
        label=r"$\mathbf{MUB\ Ensemble\ (Weighted)}$: $|\psi_k^j\rangle = U(j)|k\rangle$",
        zorder=4
    )
    if mub_err is not None and np.any(mub_err > 0):
        mub_lower = np.maximum(mub_var - 1.96 * mub_err, mub_var * 0.5)
        mub_upper = mub_var + 1.96 * mub_err
        ax.fill_between(
            qubits,
            mub_lower,
            mub_upper,
            color="#1E88E5",
            alpha=0.28,
            label=r"$\mathrm{MUB\ 95\%\ CI\ (\pm 1.96\ \mathrm{SE})}$",
            zorder=2
        )

    # Plot Haar curve and 95% confidence band
    ax.plot(
        qubits,
        haar_var,
        marker="s",
        markersize=7,
        linewidth=2.2,
        color="#D81B60",
        linestyle="--",
        label=r"$\mathbf{Haar\ Random}$: $|0\dots0\rangle, \, \vec{\theta} \sim [0, 2\pi)^P$",
        zorder=3
    )
    if haar_err is not None and np.any(haar_err > 0):
        haar_lower = np.maximum(haar_var - 1.96 * haar_err, haar_var * 0.5)
        haar_upper = haar_var + 1.96 * haar_err
        ax.fill_between(
            qubits,
            haar_lower,
            haar_upper,
            color="#D81B60",
            alpha=0.25,
            label=r"$\mathrm{Haar\ 95\%\ CI\ (\pm 1.96\ \mathrm{SE})}$",
            zorder=1
        )

    is_mean_var = results.get("metric", "") == "mean_param_var"
    is_norm_sq = results.get("metric", "") == "grad_norm_sq"
    ham_title = results.get("hamiltonian_name", "1D Model")
    lf = results.get("layers_factor", 2)

    ax.set_yscale("log")
    ax.set_xlabel(r"Number of Qubits ($n$)", fontsize=13, fontweight="bold")

    depth_str = f"L = {lf}n" if lf != 1 else "L = n"

    if is_mean_var:
        ax.set_ylabel(r"Mean Cost Derivative Variance $(1/P) \sum \mathrm{Var}[\partial_k C]$", fontsize=12, fontweight="bold")
        ax.set_title(
            f"VQE Mean Cost Derivative Variance vs. System Size $n$\n"
            f"{ham_title}\n"
            f"HEA Depth ${depth_str}$, $(1/P) \\sum \\mathrm{{Var}}[\\partial_k C]$ ($N={results.get('num_samples', 1000)}$ samples)",
            fontsize=11,
            pad=10
        )
    elif is_norm_sq:
        ax.set_ylabel(r"Gradient Norm Squared Variance $\mathrm{Var}\left[\|\nabla C\|^2\right]$", fontsize=12, fontweight="bold")
        ax.set_title(
            f"VQE Squared Gradient Norm Variance vs. System Size $n$\n"
            f"{ham_title}\n"
            f"HEA Depth ${depth_str}$, $\\mathrm{{Var}}[\\|\\nabla C\\|^2]$ ($N={results.get('num_samples', 1000)}$ samples)",
            fontsize=11,
            pad=10
        )
    else:
        ax.set_ylabel(r"Cost Derivative Variance $\mathrm{Var}[\partial_k C]$", fontsize=13, fontweight="bold")
        p_spec = str(results.get("param_idx_spec", "0"))
        if p_spec.lower() in ["mid", "middle"]:
            param_label = r"\partial_{\mathrm{mid}} C \ (\mathrm{Layer\ } n, \ \mathrm{Wire\ } \lfloor n/2 \rfloor)"
        else:
            param_label = rf"\partial_{{{p_spec}}} C"

        ax.set_title(
            f"VQE Cost Derivative Variance vs. System Size $n$\n"
            f"{ham_title}\n"
            f"HEA Depth ${depth_str}$, Parameter ${param_label}$",
            fontsize=11,
            pad=10
        )

    # Cut empty parts of y-axis and focus tightly on the data range
    all_vals_list = [haar_var, mub_var]
    if haar_err is not None and np.any(haar_err > 0):
        all_vals_list.append(np.maximum(haar_var - 1.96 * haar_err, haar_var * 0.5))
        all_vals_list.append(haar_var + 1.96 * haar_err)
    if mub_err is not None and np.any(mub_err > 0):
        all_vals_list.append(np.maximum(mub_var - 1.96 * mub_err, mub_var * 0.5))
        all_vals_list.append(mub_var + 1.96 * mub_err)
    all_vals = np.concatenate(all_vals_list)
    pos_vals = all_vals[all_vals > 0]
    if len(pos_vals) > 0:
        min_val = float(np.min(pos_vals))
        max_val = float(np.max(pos_vals))
        ax.set_ylim(bottom=min_val * 0.85, top=max_val * 1.20)
        if max_val / min_val <= 10:
            from matplotlib.ticker import ScalarFormatter
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter())

    ax.set_xticks(qubits)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=11, frameon=True, loc="best")

    plt.tight_layout()
    fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)

    return output_path
