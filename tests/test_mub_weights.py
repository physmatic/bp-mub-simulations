"""Unit tests for Hamiltonian-to-MUB weight mapping."""

import sys
from pathlib import Path
import numpy as np
import pennylane as qml

# Ensure project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "MUB_Circuits_generator"))

from src.MUB_Circuits_generator.pauli_to_j import p_to_j
from src.MUB_Circuits_generator.mub_weights import compute_mub_weights, op_to_pauli_str


def test_op_to_pauli_str():
    """Verify conversion of PennyLane operators to big-endian Pauli strings."""
    wire_order = [0, 1, 2]

    assert op_to_pauli_str(qml.PauliX(0), wire_order) == "XII"
    assert op_to_pauli_str(qml.PauliY(1), wire_order) == "IYI"
    assert op_to_pauli_str(qml.PauliZ(2), wire_order) == "IIZ"
    assert op_to_pauli_str(qml.PauliZ(0) @ qml.PauliX(2), wire_order) == "ZIX"
    assert op_to_pauli_str(qml.Identity(0) @ qml.Identity(1), wire_order) == "III"


def test_toy_hamiltonian_weights_sum_to_one():
    """Verify toy Hamiltonian H = 0.5 Z_0 Z_1 - 0.3 X_1 + 1.2 X_0 X_1 weights sum to 1.0."""
    H = 0.5 * qml.PauliZ(0) @ qml.PauliZ(1) - 0.3 * qml.PauliX(1) + 1.2 * qml.PauliX(0) @ qml.PauliX(1)
    weights = compute_mub_weights(H, p_to_j)

    total = sum(weights.values())
    assert np.isclose(total, 1.0), f"Weights do not sum to 1.0: {total}"

    j_zz = p_to_j("ZZ")  # Basis inf (computational)
    j_xx = p_to_j("XX")  # Basis 0 (X-basis)
    j_ix = p_to_j("IX")  # Basis 0 (X-basis)

    assert j_xx == j_ix == 0, f"'XX' and 'IX' should map to Clifford MUB basis 0: {j_xx} vs {j_ix}"
    assert np.isinf(j_zz), f"'ZZ' should map to computational basis inf: {j_zz}"
    assert np.isclose(weights[j_zz], 0.25), f"Expected 0.25 for Basis {j_zz}, got {weights[j_zz]}"
    assert np.isclose(weights[j_xx], 0.75), f"Expected 0.75 for Basis {j_xx}, got {weights[j_xx]}"


def test_identity_filtering():
    """Pure identity terms must be filtered out and not affect weights."""
    H1 = 0.5 * qml.PauliZ(0) @ qml.PauliZ(1) - 0.3 * qml.PauliX(1) + 1.2 * qml.PauliX(0) @ qml.PauliX(1)
    H2 = H1 + 3.5 * qml.Identity(0) @ qml.Identity(1)

    w1 = compute_mub_weights(H1, p_to_j)
    w2 = compute_mub_weights(H2, p_to_j)

    assert w1 == w2, f"Identity shifted weights: {w1} != {w2}"


def test_three_qubit_hamiltonian():
    """Verify 3-qubit Hamiltonian weights computation and normalization."""
    # H = 1.0 Z0 Z1 + 2.0 X1 X2 + 0.5 Y0 Y1 Y2
    H = (
        1.0 * qml.PauliZ(0) @ qml.PauliZ(1)
        + 2.0 * qml.PauliX(1) @ qml.PauliX(2)
        + 0.5 * qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(2)
    )
    wires = [0, 1, 2]
    weights = compute_mub_weights(H, p_to_j, wires=wires)

    assert np.isclose(sum(weights.values()), 1.0)
    w_sum = 1.0 + 2.0 + 0.5

    # Strings: 'ZZI', 'IXX', 'YYY'
    j_zzi = p_to_j("ZZI")
    j_ixx = p_to_j("IXX")
    j_yyy = p_to_j("YYY")

    # If bases are distinct or merged
    expected_weights = {}
    for j, c in [(j_zzi, 1.0), (j_ixx, 2.0), (j_yyy, 0.5)]:
        expected_weights[j] = expected_weights.get(j, 0.0) + c / w_sum

    for j, exp_w in expected_weights.items():
        assert np.isclose(weights[j], exp_w)


def test_two_qubit_multiple_paulis_same_basis():
    """
    Test H = ZZ + XX + XI + YZ.
    Both 'XX' and 'XI' belong to the same stabilizer basis (Basis 0).
    Their weights must accumulate: |1.0| + |1.0| = 2.0 out of 4.0 total.
    """
    H = (
        1.0 * qml.PauliZ(0) @ qml.PauliZ(1)
        + 1.0 * qml.PauliX(0) @ qml.PauliX(1)
        + 1.0 * qml.PauliX(0)
        + 1.0 * qml.PauliY(0) @ qml.PauliZ(1)
    )
    weights = compute_mub_weights(H, p_to_j)

    j_zz = p_to_j("ZZ")  # Basis inf
    j_xx = p_to_j("XX")  # Basis 0
    j_xi = p_to_j("XI")  # Basis 0
    j_yz = p_to_j("YZ")  # Basis 2

    assert j_xx == j_xi == 0, f"'XX' and 'XI' should map to the same basis: {j_xx} vs {j_xi}"
    assert np.isinf(j_zz), f"'ZZ' should map to inf: {j_zz}"

    # Expected: W(Basis inf) = 1.0, W(Basis 0) = 2.0, W(Basis 2) = 1.0 -> total = 4.0
    assert np.isclose(weights[j_zz], 0.25)
    assert np.isclose(weights[j_xx], 0.50)
    assert np.isclose(weights[j_yz], 0.25)
    assert np.isclose(sum(weights.values()), 1.0)


def test_three_qubit_tfim_multiple_paulis_per_basis():
    """
    Test 3-qubit TFIM: H = J*(Z0 Z1 + Z1 Z2 + Z0 Z2) + h*(X0 + X1 + X2).
    Three terms belong to Basis inf (Z-basis), three terms belong to Basis 0 (X-basis).
    """
    J = 1.5
    h = 0.8
    H = (
        J * (qml.PauliZ(0) @ qml.PauliZ(1) + qml.PauliZ(1) @ qml.PauliZ(2) + qml.PauliZ(0) @ qml.PauliZ(2))
        + h * (qml.PauliX(0) + qml.PauliX(1) + qml.PauliX(2))
    )
    weights = compute_mub_weights(H, p_to_j)

    j_z = p_to_j("ZZI")  # Basis inf
    j_x = p_to_j("XII")  # Basis 0

    assert np.isinf(j_z), f"ZZI should map to inf, got {j_z}"
    assert j_x == 0, f"XII should map to 0, got {j_x}"

    # Verify all 3 Z-terms map to j_z and all 3 X-terms map to j_x
    for p_z in ["ZZI", "IZZ", "ZIZ"]:
        assert p_to_j(p_z) == j_z, f"{p_z} should map to Z-basis {j_z}"
    for p_x in ["XII", "IXI", "IIX"]:
        assert p_to_j(p_x) == j_x, f"{p_x} should map to X-basis {j_x}"

    W_z = 3 * J
    W_x = 3 * h
    W_tot = W_z + W_x

    assert np.isclose(weights[j_z], W_z / W_tot)
    assert np.isclose(weights[j_x], W_x / W_tot)
    assert np.isclose(sum(weights.values()), 1.0)



if __name__ == "__main__":
    test_op_to_pauli_str()
    test_toy_hamiltonian_weights_sum_to_one()
    test_identity_filtering()
    test_three_qubit_hamiltonian()
    test_two_qubit_multiple_paulis_same_basis()
    test_three_qubit_tfim_multiple_paulis_per_basis()
    print("All unit tests (including multi-Pauli accumulation) passed successfully!")

