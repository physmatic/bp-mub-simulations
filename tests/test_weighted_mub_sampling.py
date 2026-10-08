"""Unit tests for Hamiltonian-weighted MUB sampling.

Hamiltonian contains all stabilizers that are NOT only X and NOT only Z,
with equal coefficients (up to sign):

    H = (
          1.0 * (qml.PauliX(wires[0]) @ qml.PauliZ(wires[1]))   # XZ -> Basis 1
        - 1.0 * (qml.PauliY(wires[0]) @ qml.PauliX(wires[1]))   # YX -> Basis 1
        + 1.0 * (qml.PauliZ(wires[0]) @ qml.PauliY(wires[1]))   # ZY -> Basis 1
        + 1.0 * (qml.PauliX(wires[0]) @ qml.PauliY(wires[1]))   # XY -> Basis 2
        - 1.0 * (qml.PauliY(wires[0]) @ qml.PauliZ(wires[1]))   # YZ -> Basis 2
        + 1.0 * (qml.PauliZ(wires[0]) @ qml.PauliX(wires[1]))   # ZX -> Basis 2
        + 1.0 * qml.PauliY(wires[0])                            # YI -> Basis 3
        - 1.0 * qml.PauliY(wires[1])                            # IY -> Basis 3
        + 1.0 * (qml.PauliY(wires[0]) @ qml.PauliY(wires[1]))   # YY -> Basis 3
    )

All 9 non-X, non-Z stabilizers are included with equal coefficient magnitude |c| = 1.0:
  - Basis 1: {XZ, YX, ZY} -> 3 terms -> w_1 = 3/9 = 1/3
  - Basis 2: {XY, YZ, ZX} -> 3 terms -> w_2 = 3/9 = 1/3
  - Basis 3: {YI, IY, YY} -> 3 terms -> w_3 = 3/9 = 1/3
  - Basis 0 (pure X): 0 terms        -> w_0 = 0/9 = 0
  - Basis inf (pure Z): 0 terms      -> w_inf = 0/9 = 0
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pennylane as qml

# Ensure project modules are importable

from src.pauli_to_j import p_to_j
from src.mub_weights import compute_mub_weights, format_basis_name, op_to_pauli_str
from src.state_preparation import (
    sample_mub_basis_and_state,
    get_mub_statevector,
)


def sample_and_print_mub_state(
    weights: Dict[Union[int, float], float],
    n: int,
    sample_idx: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Union[int, float], int, np.ndarray]:
    """
    Samples (j, i) according to MUB weights and prints the sampled basis and state index.
    
    Returns:
        j: sampled basis identifier (0..2^n-1 or float('inf'))
        i: sampled state index within basis (0 <= i < 2^n)
        state_vec: exact complex statevector |psi_i^j>
    """
    # Sample basis j ~ w_j and state index i ~ Uniform(0, 2^n - 1)
    j, i = sample_mub_basis_and_state(weights, n, rng=rng)

    # Analytical statevector |psi_i^j>
    state_vec = get_mub_statevector(n, j, i)

    j_formatted = format_basis_name(j)
    basis_type = f"Clifford MUB j={j}"
    
    # Format state vector string
    sv_rounded = [
        f"{val.real:+.2f}{val.imag:+.2f}j" if abs(val.imag) > 1e-6 else f"{val.real:+.2f}"
        for val in state_vec
    ]
    sv_str = "[" + ", ".join(sv_rounded) + "]"

    print(f"Sample {sample_idx:3d} | Sampled (j, i) = ({j_formatted:>2}, {i}) | Basis: {basis_type:<24} | State |psi_{i}^{j_formatted}> = {sv_str}")

    return j, i, state_vec


def test_weighted_mub_sampling_non_x_non_z(num_print_samples: int = 100, num_stat_samples: int = 10000):
    n_qubits = 2
    wires = [0, 1]  # Qubit 1 and Qubit 2

    # Define Hamiltonian with custom coefficients
    H = (
        + 1.0 * (qml.PauliX(wires[0]) @ qml.PauliZ(wires[1]))   # XZ (Basis 1)
        - 4.0 * (qml.PauliY(wires[0]) @ qml.PauliX(wires[1]))  # YX (Basis 1)
        + 1.0 * (qml.PauliZ(wires[0]) @ qml.PauliY(wires[1]))   # ZY (Basis 1)
        + 1.0 * (qml.PauliX(wires[0]) @ qml.PauliY(wires[1]))   # XY (Basis 2)
        + 1.0 * (qml.PauliY(wires[0]) @ qml.PauliZ(wires[1]))   # YZ (Basis 2)
        + 1.0 * (qml.PauliZ(wires[0]) @ qml.PauliX(wires[1]))   # ZX (Basis 2)
        + 0.0 * qml.PauliY(wires[0])                            # YI (Basis 3)
        - 1.0 * qml.PauliY(wires[1])                            # IY (Basis 3)
        + 1.0 * (qml.PauliY(wires[0]) @ qml.PauliY(wires[1]))   # YY (Basis 3)
    )

    print("=" * 95)
    print("WEIGHTED MUB SAMPLING TEST WITH DYNAMIC THEORETICAL WEIGHTS")
    print("=" * 95)

    # 1. Compute theoretical weights directly from Hamiltonian terms
    coeffs, ops = H.terms()
    basis_raw_weights = {}
    print("\n[Step 1] Hamiltonian Pauli Terms and Basis Mapping:")
    print("-" * 95)
    for c, op in zip(coeffs, ops):
        pauli_str = op_to_pauli_str(op, wires)
        b_idx = p_to_j(pauli_str)
        basis_raw_weights[b_idx] = basis_raw_weights.get(b_idx, 0.0) + abs(c)
        print(f"  Coeff: {c:+6.1f} | Term: {pauli_str} -> Basis {format_basis_name(b_idx):<4} (weight accumulated: +{abs(c):.1f})")

    total_weight = sum(basis_raw_weights.values())
    expected_weights = {b: w / total_weight for b, w in basis_raw_weights.items()}

    print("-" * 95)
    print(f"  Total weight sum W = {total_weight:.2f}")
    print(f"  Theoretical expected weights:")
    for b in sorted(expected_weights.keys(), key=lambda x: (np.isinf(x), x)):
        print(f"    - Basis {format_basis_name(b)}: W={basis_raw_weights[b]:.1f}/{total_weight:.1f} = {expected_weights[b]:.4f} ({expected_weights[b]*100:.1f}%)")

    # 2. Compute weights via compute_mub_weights function and assert equivalence
    weights = compute_mub_weights(H, p_to_j, wires=wires)
    print(f"\n  Computed weights from compute_mub_weights: { {format_basis_name(k): round(v, 6) for k, v in weights.items()} }")
    print("-" * 95)

    for b, exp_w in expected_weights.items():
        assert np.isclose(weights[b], exp_w), f"Basis {b} weight mismatch: {weights[b]} vs {exp_w}"

    # 3. Sampling loop with detailed per-sample prints
    print(f"\n[Step 2] Executing {num_print_samples} sampling draws and printing each (j, i) for |psi_i^j>:")
    print("-" * 95)

    rng = np.random.default_rng(2026)
    active_bases = sorted(weights.keys(), key=lambda x: (np.isinf(x), x))
    sample_counts = {b: {i: 0 for i in range(4)} for b in active_bases}

    for s in range(1, num_print_samples + 1):
        j, i, _ = sample_and_print_mub_state(weights, n_qubits, sample_idx=s, rng=rng)
        sample_counts[j][i] += 1

    # 4. Summary of the printed batch
    print("-" * 95)
    print(f"Summary of {num_print_samples} printed samples:")
    for b in active_bases:
        c = sum(sample_counts[b].values())
        exp_pct = expected_weights[b] * 100
        obs_pct = c / num_print_samples * 100
        print(f"  - Basis j={format_basis_name(b)}: sampled {c:3d}/{num_print_samples} times ({obs_pct:5.1f}%) [Theoretical: {exp_pct:5.1f}%]")
        print(f"      State indices breakdown: { {f'i={k}': sample_counts[b][k] for k in range(4)} }")

    # 5. Statistical convergence test over larger sample size
    print(f"\n[Step 3] Statistical convergence validation over {num_stat_samples:,} samples:")
    stat_counts_j = {b: 0 for b in active_bases}
    stat_counts_i = {0: 0, 1: 0, 2: 0, 3: 0}

    for _ in range(num_stat_samples):
        j, i = sample_mub_basis_and_state(weights, n_qubits, rng=rng)
        stat_counts_j[j] += 1
        stat_counts_i[i] += 1

    for b in active_bases:
        freq_b = stat_counts_j[b] / num_stat_samples
        exp_w = expected_weights[b]
        print(f"  - Observed w_{format_basis_name(b)}: {freq_b:.4f} (Expected: {exp_w:.4f}, error: {abs(freq_b - exp_w):.4f})")
        assert np.isclose(freq_b, exp_w, atol=0.02), f"Basis {b} frequency deviated: {freq_b} vs {exp_w}"

    for i in range(4):
        freq_i = stat_counts_i[i] / num_stat_samples
        print(f"  - State index i={i} frequency: {freq_i:.4f} (Expected: 0.2500, error: {abs(freq_i - 0.25):.4f})")
        assert np.isclose(freq_i, 0.25, atol=0.015)

    print("\n" + "=" * 95)
    print("ALL WEIGHTED MUB SAMPLING CHECKS PASSED SUCCESSFULLY!")
    print("=" * 95)


if __name__ == "__main__":
    test_weighted_mub_sampling_non_x_non_z(num_print_samples=100, num_stat_samples=10000)
