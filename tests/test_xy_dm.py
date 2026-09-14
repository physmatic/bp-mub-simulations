"""Unit tests for the 1D Transverse-Field XY Model with DM interaction and stabilizer counting."""

import sys
from pathlib import Path
import numpy as np
import pennylane as qml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "MUB_Circuits_generator"))

from src.MUB_Circuits_generator.hamiltonians import build_tfim_hamiltonian, build_xy_dm_hamiltonian
from src.MUB_Circuits_generator.mub_weights import (
    compute_mub_weights,
    count_active_stabilizer_sets,
    get_active_stabilizer_sets,
    format_basis_name,
)
from src.MUB_Circuits_generator.pauli_to_j import p_to_j


def test_xy_dm_term_counts():
    """Verify term counts for open-chain XY-DM Hamiltonian: 4*(n-1) + n = 5n - 4 terms."""
    for n in [2, 3, 4, 5, 6]:
        H = build_xy_dm_hamiltonian(n, Jx=1.0, Jy=0.5, D=0.8, h=1.0)
        coeffs, ops = H.terms()
        expected = 5 * n - 4
        assert len(ops) == expected, f"n={n}: expected {expected} terms, got {len(ops)}"
        assert len(coeffs) == expected


def test_count_active_stabilizers_tfim():
    """TFIM only activates exactly 2 stabilizer sets (computational ∞ and Hadamard 0)."""
    for n in [2, 3, 4, 5, 6]:
        H = build_tfim_hamiltonian(n, J=1.0, h=1.0)
        num_sets = count_active_stabilizer_sets(H, p_to_j)
        assert num_sets == 2, f"TFIM n={n} must have exactly 2 active stabilizer sets, got {num_sets}"


def test_count_active_stabilizers_xy_dm():
    """
    Verify the XY-DM model activates multiple entangled MUB stabilizer sets
    across n=3, 4, 5, 6.
    """
    expected_counts = {
        3: 7,
        4: 8,
        5: 11,
        6: 13,
    }

    print("\n--- Active Stabilizers in 1D XY-DM Model ---")
    for n in [3, 4, 5, 6]:
        H = build_xy_dm_hamiltonian(n, Jx=1.0, Jy=0.5, D=0.8, h=1.0)
        num_sets = count_active_stabilizer_sets(H, p_to_j)
        active_sets = get_active_stabilizer_sets(H, p_to_j)
        formatted_bases = [format_basis_name(k) for k in active_sets.keys()]

        print(f"n={n}: {num_sets} active stabilizer sets (out of {2**n + 1} total bases)")
        print(f"      Active bases: {formatted_bases}")

        assert num_sets == expected_counts[n], (
            f"n={n}: expected {expected_counts[n]} active sets, got {num_sets}"
        )
        assert np.isclose(sum(active_sets.values()), 1.0)


def test_xy_dm_coefficient_signs():
    """Verify coefficient values and signs for n=2 XY-DM Hamiltonian."""
    n = 2
    Jx, Jy, D, h = 1.0, 0.5, 0.8, 1.0
    H = build_xy_dm_hamiltonian(n, Jx=Jx, Jy=Jy, D=D, h=h)
    coeffs, ops = H.terms()

    # Expected:
    # -Jx * X0 X1 = -1.0
    # -Jy * Y0 Y1 = -0.5
    # -D * X0 Y1  = -0.8
    # +D * Y0 X1  = +0.8
    # -h * Z0     = -1.0
    # -h * Z1     = -1.0
    assert len(coeffs) == 6
    expected_coeffs = [-1.0, -0.5, -0.8, 0.8, -1.0, -1.0]
    np.testing.assert_allclose(coeffs, expected_coeffs, atol=1e-7)


if __name__ == "__main__":
    test_xy_dm_term_counts()
    test_count_active_stabilizers_tfim()
    test_count_active_stabilizers_xy_dm()
    test_xy_dm_coefficient_signs()
    print("\nAll XY-DM and active stabilizer tests passed successfully!")
