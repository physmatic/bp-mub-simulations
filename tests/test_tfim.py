"""Unit tests for the 1D TFIM Hamiltonian generator and MUB weight integration."""

import numpy as np
import pennylane as qml

# Ensure project modules are importable

from mub.hamiltonians import build_tfim_hamiltonian
from mub.mub_weights import compute_mub_weights, format_basis_name
from mub.pauli_to_j import p_to_j


def test_tfim_term_count():
    """Verify open-chain TFIM has exactly (n - 1) ZZ terms + n X terms = (2n - 1) terms."""
    for n in [2, 3, 4, 5]:
        H = build_tfim_hamiltonian(n, J=1.0, h=1.0)
        coeffs, ops = H.terms()
        expected_terms = (n - 1) + n
        assert len(ops) == expected_terms, f"n={n}: expected {expected_terms} terms, got {len(ops)}"
        assert len(coeffs) == expected_terms


def test_tfim_n3_weights_and_basis_distribution():
    """
    Verify n = 3 open-chain TFIM with J = 1.0, h = 1.0:
      - Has exactly (3 - 1) + 3 = 5 Pauli terms.
      - Has non-zero weights on both Z-type (Basis ∞) and X-type (Basis 0) stabilizer bases.
      - Exact weights: w_∞ = 2/5 = 0.4, w_0 = 3/5 = 0.6.
    """
    n = 3
    J = 1.0
    h = 1.0
    H = build_tfim_hamiltonian(n, J=J, h=h)

    coeffs, ops = H.terms()
    assert len(ops) == 5, f"Expected exactly 5 terms for n=3, got {len(ops)}"

    # Compute MUB weights
    weights = compute_mub_weights(H, p_to_j)

    # 1. Non-zero weights on both Z-type and X-type bases
    assert float("inf") in weights, "Z-type basis (∞) must be present"
    assert 0 in weights, "X-type basis (0) must be present"
    assert weights[float("inf")] > 0.0, "Z-type basis weight must be > 0"
    assert weights[0] > 0.0, "X-type basis weight must be > 0"

    # 2. Normalization: sum to 1.0
    assert np.isclose(sum(weights.values()), 1.0), "Weights must sum to 1.0"

    # 3. Exact theoretical weight values:
    # W_Z = 2 * |-1.0| = 2.0 (from Z0 Z1, Z1 Z2)
    # W_X = 3 * |-1.0| = 3.0 (from X0, X1, X2)
    # Total W = 5.0
    # w_∞ = 2.0 / 5.0 = 0.40
    # w_0 = 3.0 / 5.0 = 0.60
    assert np.isclose(weights[float("inf")], 0.40), f"Expected 0.40 for Basis ∞, got {weights[float('inf')]}"
    assert np.isclose(weights[0], 0.60), f"Expected 0.60 for Basis 0, got {weights[0]}"

    readable = {format_basis_name(k): v for k, v in weights.items()}
    print(f"n=3 TFIM MUB weights: {readable}")


def test_tfim_custom_parameters():
    """Verify weights with non-uniform J and h parameters."""
    n = 4
    J = 2.5
    h = 0.5
    H = build_tfim_hamiltonian(n, J=J, h=h)

    weights = compute_mub_weights(H, p_to_j)

    # (4 - 1) ZZ terms with |c| = 2.5 -> W_Z = 3 * 2.5 = 7.5
    # 4 X terms with |c| = 0.5       -> W_X = 4 * 0.5 = 2.0
    # W_tot = 9.5
    w_z_expected = 7.5 / 9.5
    w_x_expected = 2.0 / 9.5

    assert np.isclose(weights[float("inf")], w_z_expected)
    assert np.isclose(weights[0], w_x_expected)
    assert np.isclose(sum(weights.values()), 1.0)


if __name__ == "__main__":
    test_tfim_term_count()
    test_tfim_n3_weights_and_basis_distribution()
    test_tfim_custom_parameters()
    print("All TFIM generator tests passed successfully!")
