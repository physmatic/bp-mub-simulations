"""
Hamiltonian-Weighted MUB Ensembles.

Calculates the basis probability distribution w_j over the (2^n + 1)
mutually unbiased bases based on the sum of absolute values of active
Pauli coefficients in a Hamiltonian.
"""


from typing import Callable, Dict, Optional, Sequence, Union
import numpy as np
import pennylane as qml


def op_to_pauli_str(op: qml.operation.Operator, wire_order: Sequence) -> str:
    """
    Translates a PennyLane Pauli operator into a Pauli string across `wire_order`.
    Follows PennyLane's big-endian convention: index 0 (leftmost char)
    corresponds to wire_order[0].
    """
    wire_map = {w: 'I' for w in wire_order}

    # 1. Inspect op.pauli_rep if available
    if hasattr(op, 'pauli_rep') and op.pauli_rep is not None:
        for pw in op.pauli_rep.keys():
            for w, p in pw.items():
                if w in wire_map:
                    wire_map[w] = p
    else:
        # 2. Recursive decomposition for composite operators
        def _extract(single_op):
            name = single_op.name
            if name in ['PauliX', 'X']:
                wire_map[single_op.wires[0]] = 'X'
            elif name in ['PauliY', 'Y']:
                wire_map[single_op.wires[0]] = 'Y'
            elif name in ['PauliZ', 'Z']:
                wire_map[single_op.wires[0]] = 'Z'
            elif name in ['Identity', 'I']:
                pass
            elif hasattr(single_op, 'operands'):
                for sub in single_op.operands:
                    _extract(sub)

        _extract(op)

    return "".join(wire_map[w] for w in wire_order)


def compute_mub_weights(
    H: Union[qml.Hamiltonian, qml.operation.Operator],
    p_to_j_fn: Callable[[str], int],
    wires: Optional[Sequence] = None
) -> Dict[int, float]:
    """
    Computes the normalized probability distribution w_j over MUB stabilizer bases
    for a given PennyLane Hamiltonian.

    Parameters
    ----------
    H : qml.Hamiltonian or qml.operation.Operator
        The target Hamiltonian.
    p_to_j_fn : Callable[[str], int]
        Mapping function that sends a non-identity Pauli string to its unique
        stabilizer basis index j in {1, ..., 2^n + 1}.
    wires : Sequence, optional
        Ordered sequence of wires representing the system. If None, inferred
        from H.wires (sorted integer order if all wires are integers).

    Returns
    -------
    dict[int, float]
        Normalized probability distribution {basis_index: weight}, where sum(w_j) == 1.0.
    """
    # 1. Infer wire ordering
    if wires is None:
        h_wires = list(H.wires)
        if all(isinstance(w, (int, np.integer)) for w in h_wires):
            max_wire = max(h_wires) if h_wires else 0
            wire_order = list(range(max_wire + 1))
        else:
            wire_order = sorted(h_wires)
    else:
        wire_order = list(wires)

    # 2. Extract Hamiltonian terms: H = sum c_P * P
    if hasattr(H, "terms"):
        coeffs, ops = H.terms()
    else:
        # Single operator or sum
        coeffs = [1.0]
        ops = [H]

    # 3. Accumulate raw weights W_j = sum_{P, p_to_j(P)=j} |c_P|
    raw_weights: Dict[int, float] = {}

    for coeff, op in zip(coeffs, ops):
        pauli_str = op_to_pauli_str(op, wire_order)

        # Filter out pure identity terms (P = I^{\otimes n})
        if set(pauli_str) == {'I'}:
            continue

        weight = float(abs(coeff))
        if weight == 0.0:
            continue

        j = p_to_j_fn(pauli_str)
        raw_weights[j] = raw_weights.get(j, 0.0) + weight

    # 4. Normalize distribution: w_j = W_j / sum_k W_k
    total_weight = sum(raw_weights.values())
    if total_weight == 0.0:
        return {}

    normalized_weights = {j: w / total_weight for j, w in raw_weights.items()}
    return normalized_weights


def format_basis_name(j: Union[int, float]) -> str:
    """Formats basis index j as a human-readable string, displaying '∞' for computational basis."""
    return "∞" if (j == float('inf') or np.isinf(j)) else str(j)


def test_toy_hamiltonian():
    """Unit test on a 2-qubit toy Hamiltonian: H = 0.5 Z_0 Z_1 - 0.3 X_1 + 1.2 X_0 X_1."""
    try:
        from .pauli_to_j import p_to_j
    except ImportError:
        from pauli_to_j import p_to_j

    # 1. Define toy Hamiltonian
    H = 0.5 * qml.PauliZ(0) @ qml.PauliZ(1) - 0.3 * qml.PauliX(1) + 1.2 * qml.PauliX(0) @ qml.PauliX(1)

    # 2. Compute MUB weights
    weights = compute_mub_weights(H, p_to_j)

    # 3. Validations
    readable_weights = {format_basis_name(k): v for k, v in weights.items()}
    print("Computed weights:", readable_weights)
    total_weight = sum(weights.values())
    print("Sum of weights:", total_weight)

    assert np.isclose(total_weight, 1.0), f"Weights do not sum to 1.0 (got {total_weight})"

    # Expected values using sum of absolute values (Clifford / math convention):
    # c1 = 0.5 (Z0 Z1 -> 'ZZ', Basis j=∞)
    # c2 = -0.3 (X1 -> 'IX', Basis j=0)
    # c3 = 1.2 (X0 X1 -> 'XX', Basis j=0)
    # W_total = 0.5 + 0.3 + 1.2 = 2.0
    # W(Basis ∞) = 0.5 -> w_∞ = 0.5 / 2.0 = 0.25
    # W(Basis 0) = 0.3 + 1.2 = 1.5 -> w_0 = 1.5 / 2.0 = 0.75
    j_zz = p_to_j("ZZ")  # float('inf')
    j_xx = p_to_j("XX")  # 0
    j_ix = p_to_j("IX")  # 0

    assert j_xx == j_ix == 0, f"'XX' and 'IX' should map to Clifford MUB basis 0: {j_xx} vs {j_ix}"
    assert np.isinf(j_zz), f"'ZZ' should map to computational basis ∞: {j_zz}"
    assert np.isclose(weights[j_zz], 0.25), f"Expected 0.25 for Basis ∞, got {weights[j_zz]}"
    assert np.isclose(weights[j_xx], 0.75), f"Expected 0.75 for Basis 0, got {weights[j_xx]}"

    print(f"✓ Unit test passed! Basis ∞ (ZZ): {weights[j_zz]:.4f}, Basis 0 (IX + XX): {weights[j_xx]:.4f}")






if __name__ == '__main__':
    test_toy_hamiltonian()
