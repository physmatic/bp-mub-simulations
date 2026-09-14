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
    from .hamiltonians import build_tfim_hamiltonian, build_xy_dm_hamiltonian
    from .mub_weights import compute_mub_weights, count_active_stabilizer_sets
    from .pauli_to_j import p_to_j
    from .state_preparation import sample_mub_basis_and_state, prepare_mub_state
except ImportError:
    from ansatz import hardware_efficient_ansatz, get_num_hea_params
    from hamiltonians import build_tfim_hamiltonian, build_xy_dm_hamiltonian
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


def run_variance_benchmark(
    qubits_list: Sequence[int] = (2, 3, 4, 5, 6),
    num_samples: int = 200,
    hamiltonian_type: Union[str, Callable[[int], qml.Hamiltonian]] = "xy_dm",
    layers_factor: int = 2,
    layers_fn: Optional[Callable[[int], int]] = None,
    param_idx: Union[int, str, Callable[[int, int], int]] = "mid",
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Union[List[int], List[float], Dict]]:
    """
    Sweeps over qubit counts and computes empirical variance of cost function
    derivatives Var[∂_k C] for both Haar initialization and Hamiltonian-Weighted
    MUB Ensemble initialization.

    Parameters
    ----------
    qubits_list : Sequence[int]
        List of qubit counts to benchmark, e.g. [2, 3, 4, 5, 6].
    num_samples : int
        Number of random initialization trials per qubit count.
    hamiltonian_type : str or Callable, default="xy_dm"
        Hamiltonian to benchmark: "xy_dm" (1D XY with DM interaction) or "tfim",
        or a custom callable n -> qml.Hamiltonian.
    layers_factor : int, default=2
        Multiplier for ansatz depth: layers L = layers_factor * n.
    layers_fn : Callable[[int], int], optional
        Custom function n -> layers. Overrides layers_factor if provided.
    param_idx : int, str, or Callable, default="mid"
        Parameter index k for which ∂_k C is evaluated.
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
        - "param_idx_spec": specification passed for param_idx
        - "evaluated_param_indices": dict mapping n to actual k evaluated
        - "haar_var": list of Var_Haar[∂_k C]
        - "haar_mean": list of Mean_Haar[∂_k C]
        - "mub_var": list of Var_MUB[∂_k C]
        - "mub_mean": list of Mean_MUB[∂_k C]
        - "details": dict mapping n to detailed sample lists and metadata
    """
    if layers_fn is None:
        layers_fn = lambda n: layers_factor * n

    if isinstance(hamiltonian_type, str):
        if hamiltonian_type.lower() in ["xy_dm", "xy", "dm"]:
            ham_builder = lambda n, wires: build_xy_dm_hamiltonian(n, Jx=1.0, Jy=0.5, D=0.8, h=1.0, wires=wires)
            ham_name = "1D XY-DM Model ($J_x=1.0, J_y=0.5, D=0.8, h=1.0$)"
        elif hamiltonian_type.lower() == "tfim":
            ham_builder = lambda n, wires: build_tfim_hamiltonian(n, J=1.0, h=1.0, wires=wires)
            ham_name = "1D TFIM Model ($J=1.0, h=1.0$)"
        else:
            raise ValueError(f"Unknown hamiltonian_type '{hamiltonian_type}'. Choose 'xy_dm' or 'tfim'.")
    else:
        ham_builder = hamiltonian_type
        ham_name = getattr(hamiltonian_type, "__name__", "Custom Hamiltonian")

    rng = np.random.default_rng(seed)

    results = {
        "qubits": list(qubits_list),
        "num_samples": num_samples,
        "param_idx_spec": str(param_idx) if not callable(param_idx) else "callable",
        "evaluated_param_indices": {},
        "hamiltonian_name": ham_name,
        "layers_factor": layers_factor,
        "haar_var": [],
        "haar_mean": [],
        "mub_var": [],
        "mub_mean": [],
        "active_stabilizer_counts": {},
        "details": {}
    }

    if verbose:
        print("=" * 80)
        print(f" COST DERIVATIVE VARIANCE BENCHMARK: MUB ENSEMBLE vs. HAAR INITIALIZATION")
        print(f" Hamiltonian: {ham_name}")
        print(f" Ansatz Depth: L = {layers_factor}n | Samples per n: {num_samples} | Param: {param_idx}")
        print("=" * 80)
        print(f"{'n':<4} | {'L':<4} | {'Params':<6} | {'k_mid':<6} | {'Active':<7} | {'Var[∂_k C] Haar':<18} | {'Var[∂_k C] MUB':<18} | {'Time (s)':<8}")
        print("-" * 80)

    for n in qubits_list:
        t0 = time.time()
        layers = layers_fn(n)
        num_p = get_num_hea_params(n, layers)
        wires = list(range(n))

        # Determine target parameter index k
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

        H = ham_builder(n, wires)
        weights = compute_mub_weights(H, p_to_j, wires=wires)
        active_count = len(weights)
        results["active_stabilizer_counts"][n] = active_count
        dev = qml.device("default.qubit", wires=wires)

        # 1. Haar Initialization Trials
        @qml.qnode(dev)
        def haar_circuit(p):
            hardware_efficient_ansatz(p, wires=wires, layers=layers)
            return qml.expval(H)

        haar_grads = []
        for _ in range(num_samples):
            theta = rng.uniform(0.0, 2.0 * np.pi, size=num_p)
            grad_val = evaluate_single_param_shift(haar_circuit, theta, param_idx=actual_k)
            haar_grads.append(grad_val)

        haar_grads = np.array(haar_grads)
        h_var = float(np.var(haar_grads, ddof=1))
        h_mean = float(np.mean(haar_grads))

        # 2. Hamiltonian-Weighted MUB Ensemble Initialization Trials
        p0 = np.zeros(num_p, dtype=float)
        mub_grads = []
        for _ in range(num_samples):
            # Sample basis j ~ w_j and eigenstate k ~ Uniform(0, 2^n - 1)
            j, k = sample_mub_basis_and_state(weights, n, rng=rng)

            @qml.qnode(dev)
            def mub_circuit(p, _j=j, _k=k):
                prepare_mub_state(n, _j, _k, wires=wires)
                hardware_efficient_ansatz(p, wires=wires, layers=layers)
                return qml.expval(H)

            grad_val = evaluate_single_param_shift(mub_circuit, p0, param_idx=actual_k)
            mub_grads.append(grad_val)

        mub_grads = np.array(mub_grads)
        m_var = float(np.var(mub_grads, ddof=1))
        m_mean = float(np.mean(mub_grads))

        dt = time.time() - t0

        results["haar_var"].append(h_var)
        results["haar_mean"].append(h_mean)
        results["mub_var"].append(m_var)
        results["mub_mean"].append(m_mean)
        results["details"][n] = {
            "layers": layers,
            "num_params": num_p,
            "param_k": actual_k,
            "haar_grads": haar_grads.tolist(),
            "mub_grads": mub_grads.tolist(),
            "runtime_sec": dt,
        }

        if verbose:
            print(f"{n:<4} | {layers:<4} | {num_p:<6} | {actual_k:<6} | {active_count:<7} | {h_var:<18.6e} | {m_var:<18.6e} | {dt:<8.2f}")

    if verbose:
        print("=" * 80)

    return results


def plot_variance_benchmark(
    results: Dict,
    output_path: Union[str, Path] = "outputs/gradient_variance_benchmark.png",
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

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Plot empirical points and lines
    ax.plot(
        qubits,
        mub_var,
        marker="o",
        markersize=8,
        linewidth=2.2,
        color="#1E88E5",
        label=r"$\mathbf{MUB\ Ensemble\ (Weighted)}$: $|\psi_k^j\rangle = U(j)|k\rangle$",
        zorder=4
    )

    ax.plot(
        qubits,
        haar_var,
        marker="s",
        markersize=8,
        linewidth=2.2,
        color="#D81B60",
        linestyle="--",
        label=r"$\mathbf{Haar\ Random}$: $|0\dots0\rangle, \, \vec{\theta} \sim [0, 2\pi)^P$",
        zorder=3
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"Number of Qubits ($n$)", fontsize=13, fontweight="bold")
    ax.set_ylabel(r"Cost Derivative Variance $\mathrm{Var}[\partial_k C]$", fontsize=13, fontweight="bold")

    ham_title = results.get("hamiltonian_name", "1D Model")
    lf = results.get("layers_factor", 2)
    p_spec = str(results.get("param_idx_spec", "0"))

    if p_spec.lower() in ["mid", "middle"]:
        param_label = r"\partial_{\mathrm{mid}} C \ (\mathrm{Layer\ } n, \ \mathrm{Wire\ } \lfloor n/2 \rfloor)"
    else:
        param_label = rf"\partial_{{{p_spec}}} C"

    ax.set_title(
        f"VQE Cost Derivative Variance vs. System Size $n$\n"
        f"{ham_title}\n"
        f"HEA Depth $L={lf}n$, Parameter ${param_label}$",
        fontsize=11,
        pad=10
    )

    ax.set_xticks(qubits)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=11, frameon=True, loc="lower left")

    plt.tight_layout()
    fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)

    return output_path
