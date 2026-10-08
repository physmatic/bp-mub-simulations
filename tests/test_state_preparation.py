"""
Unit tests for MUB state preparation and sampling.

Tests:
1. PennyLane circuit state preparation matches analytical MUB statevectors.
2. Orthogonality of states within the same basis: <psi_k^j | psi_k'^j> = delta_{k, k'}.
3. Mutual unbiasedness across different bases: |<psi_k^j | psi_k'^{j'}>|^2 = 1/2^n.
4. Statistical correctness of sample_mub_basis_and_state.
5. Error handling for invalid eigenstate indices, wires, and empty weights.
"""

import numpy as np
import pennylane as qml


from mub import (
    sample_mub_basis_and_state,
    prepare_mub_state,
    get_mub_statevector,
    sample_and_prepare_mub_state,
)


def test_mub_state_matches_analytical_2qubit():
    """Verify prepare_mub_state produces the exact analytical statevector for n=2."""
    n = 2
    dev = qml.device("default.qubit", wires=n)

    bases = [float('inf'), 0, 1, 2, 3]

    for j in bases:
        for k in range(2**n):
            @qml.qnode(dev)
            def circuit():
                prepare_mub_state(n, j, k)
                return qml.state()

            simulated_state = circuit()
            analytical_state = get_mub_statevector(n, j, k)

            # Check direct vector equality
            np.testing.assert_allclose(
                simulated_state,
                analytical_state,
                atol=1e-7,
                err_msg=f"State mismatch for n={n}, j={j}, k={k}"
            )


def test_mub_state_matches_analytical_3qubit():
    """Verify prepare_mub_state produces the exact analytical statevector for n=3."""
    n = 3
    dev = qml.device("default.qubit", wires=n)

    bases = [float('inf'), 0, 3, 7]

    for j in bases:
        for k in [0, 2, 5, 7]:
            @qml.qnode(dev)
            def circuit():
                prepare_mub_state(n, j, k)
                return qml.state()

            simulated_state = circuit()
            analytical_state = get_mub_statevector(n, j, k)

            np.testing.assert_allclose(
                simulated_state,
                analytical_state,
                atol=1e-7,
                err_msg=f"State mismatch for n={n}, j={j}, k={k}"
            )


def test_orthogonality_within_basis():
    """States within the same MUB basis must be orthonormal: <psi_k^j | psi_k'^j> = delta_{k, k'}."""
    n = 2
    dim = 2**n
    bases = [float('inf'), 0, 1, 2, 3]

    for j in bases:
        states = [get_mub_statevector(n, j, k) for k in range(dim)]
        for k1 in range(dim):
            for k2 in range(dim):
                inner_prod = np.vdot(states[k1], states[k2])
                expected = 1.0 if k1 == k2 else 0.0
                assert np.isclose(inner_prod, expected, atol=1e-7), (
                    f"Orthonormality failed for j={j}, k1={k1}, k2={k2}: got {inner_prod}"
                )


def test_mutual_unbiasedness_across_bases():
    """Overlap magnitude between states of different bases must equal 1/sqrt(2^n): |<psi_k^j | psi_k'^{j'}>|^2 = 1/2^n."""
    n = 2
    dim = 2**n
    expected_overlap_sq = 1.0 / dim
    bases = [float('inf'), 0, 1, 2, 3]

    for i1, j1 in enumerate(bases):
        for j2 in bases[i1 + 1:]:
            for k1 in range(dim):
                for k2 in range(dim):
                    psi1 = get_mub_statevector(n, j1, k1)
                    psi2 = get_mub_statevector(n, j2, k2)
                    overlap_sq = np.abs(np.vdot(psi1, psi2)) ** 2
                    assert np.isclose(overlap_sq, expected_overlap_sq, atol=1e-7), (
                        f"MUB condition failed for j1={j1}, j2={j2}, k1={k1}, k2={k2}: got {overlap_sq}"
                    )


def test_sampling_statistics():
    """Verify sample_mub_basis_and_state draws j ~ w_j and k ~ Uniform(0, 2^n - 1)."""
    n = 2
    weights = {float('inf'): 0.4, 0: 0.6}
    num_samples = 10000
    rng = np.random.default_rng(42)

    j_counts = {float('inf'): 0, 0: 0}
    k_counts = np.zeros(2**n, dtype=int)

    for _ in range(num_samples):
        j, k = sample_mub_basis_and_state(weights, n, rng=rng)
        j_counts[j] += 1
        k_counts[k] += 1

    # Check basis distribution
    assert np.isclose(j_counts[float('inf')] / num_samples, 0.4, atol=0.02)
    assert np.isclose(j_counts[0] / num_samples, 0.6, atol=0.02)

    # Check eigenstate distribution is uniform across 4 states (25% each)
    expected_k_freq = 0.25
    for k in range(2**n):
        freq = k_counts[k] / num_samples
        assert np.isclose(freq, expected_k_freq, atol=0.02), f"k={k} frequency {freq} deviates from {expected_k_freq}"


def test_sample_and_prepare_mub_state_qnode():
    """Verify sample_and_prepare_mub_state can be called directly inside a PennyLane QNode."""
    n = 2
    dev = qml.device("default.qubit", wires=n)
    weights = {0: 1.0}

    @qml.qnode(dev)
    def circuit():
        sample_and_prepare_mub_state(weights, n)
        return qml.state()

    state = circuit()
    assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)


def test_error_handling():
    """Verify appropriate exceptions for invalid inputs."""
    n = 2
    # Invalid k
    try:
        get_mub_statevector(n, 0, 4)
        assert False, "Expected ValueError for k=4"
    except ValueError as e:
        assert "out of range" in str(e)

    try:
        get_mub_statevector(n, 0, -1)
        assert False, "Expected ValueError for k=-1"
    except ValueError as e:
        assert "out of range" in str(e)

    # Empty weights
    try:
        sample_mub_basis_and_state({}, n)
        assert False, "Expected ValueError for empty weights"
    except ValueError as e:
        assert "empty weights" in str(e)

    # Wrong wire count
    dev = qml.device("default.qubit", wires=3)
    @qml.qnode(dev)
    def circuit():
        prepare_mub_state(n, 0, 0, wires=[0])
        return qml.state()

    try:
        circuit()
        assert False, "Expected ValueError for wire mismatch"
    except ValueError as e:
        assert "Expected 2 wires" in str(e)


if __name__ == "__main__":
    test_mub_state_matches_analytical_2qubit()
    test_mub_state_matches_analytical_3qubit()
    test_orthogonality_within_basis()
    test_mutual_unbiasedness_across_bases()
    test_sampling_statistics()
    test_sample_and_prepare_mub_state_qnode()
    test_error_handling()
    print("All state preparation and sampling tests passed successfully!")
