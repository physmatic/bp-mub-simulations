"""
MUB Ensemble Sampling and Quantum State Preparation.

Provides tools to:
1. Sample a stabilizer basis j ~ w_j according to Hamiltonian-derived weights.
2. Uniformly sample an eigenstate index k ~ Uniform(0, 2^n - 1).
3. Prepare the quantum state |psi_k^j> = U(j)|k> on PennyLane quantum registers.
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pennylane as qml

try:
    from .mub_circuit import mub_circuit, mub_unitary
except ImportError:
    from mub_circuit import mub_circuit, mub_unitary


def sample_mub_basis_and_state(
    weights: Dict[Union[int, float], float],
    n: int,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Union[int, float], int]:
    """
    Samples a stabilizer basis j according to probability distribution w_j,
    and then samples an eigenstate index k uniformly within that basis:

        j ~ w_j,    k ~ Uniform({0, ..., 2^n - 1})

    Parameters
    ----------
    weights : Dict[Union[int, float], float]
        Probability distribution over bases, e.g. from compute_mub_weights.
    n : int
        Number of qubits.
    rng : np.random.Generator, optional
        NumPy random number generator for reproducible sampling.

    Returns
    -------
    j : int or float('inf')
        The sampled basis identifier (float('inf') for computational, or 0..2^n-1).
    k : int
        The sampled eigenstate index in {0, ..., 2^n - 1}.
    """
    if not weights:
        raise ValueError("Cannot sample from an empty weights dictionary.")

    if rng is None:
        rng = np.random.default_rng()

    # 1. Sample basis j ~ w_j
    bases = list(weights.keys())
    probs = np.array([weights[b] for b in bases], dtype=float)
    probs /= probs.sum()  # Guard against minor floating point drift

    sampled_idx = rng.choice(len(bases), p=probs)
    j = bases[sampled_idx]

    # 2. Sample eigenstate index k ~ Uniform(0, 2^n - 1)
    k = int(rng.integers(0, 2**n))

    return j, k


def prepare_mub_state(
    n: int,
    j: Union[int, float],
    k: int,
    wires: Optional[Sequence] = None
) -> None:
    """
    PennyLane quantum function that prepares the MUB eigenstate |psi_k^j> = U(j)|k>
    on the specified wire register:

    - Step 1: Prepares computational basis state |k> via PauliX gates on wires
      where the bit of k is 1 (wire 0 is the most significant bit).
    - Step 2: Applies Clifford change-of-basis unitary U(j) = mub_circuit(n, j).
      If j = inf (computational basis), U(inf) = Identity (no extra gates).

    Parameters
    ----------
    n : int
        Number of qubits.
    j : int or float('inf')
        Basis index. float('inf') denotes the standard computational basis;
        an integer in {0, ..., 2^n - 1} denotes the MUB Clifford parameter.
    k : int
        Eigenstate index in {0, ..., 2^n - 1}.
    wires : Sequence, optional
        Ordered wire labels. If None, defaults to list(range(n)).
    """
    if wires is None:
        wire_list = list(range(n))
    else:
        wire_list = list(wires)
        if len(wire_list) != n:
            raise ValueError(f"Expected {n} wires, got {len(wire_list)}")

    if not (0 <= k < 2**n):
        raise ValueError(f"Eigenstate index k={k} out of range for n={n} qubits.")

    # 1. Prepare |k> in big-endian order (wire 0 is most significant bit)
    k_bits = [(k >> (n - 1 - i)) & 1 for i in range(n)]
    for i, bit in enumerate(k_bits):
        if bit == 1:
            qml.PauliX(wires=wire_list[i])

    # 2. Apply Clifford change-of-basis unitary U(j)
    is_computational = (j == float('inf') or np.isinf(j))
    if not is_computational:
        mub_circuit(n, int(j), wires=wire_list)


def get_mub_statevector(
    n: int,
    j: Union[int, float],
    k: int
) -> np.ndarray:
    """
    Returns the analytical statevector |psi_k^j> = U(j)|k> as a 1D NumPy array.

    Parameters
    ----------
    n : int
        Number of qubits.
    j : int or float('inf')
        Basis index.
    k : int
        Eigenstate index in {0, ..., 2^n - 1}.

    Returns
    -------
    np.ndarray
        Complex statevector of length 2^n.
    """
    dim = 2**n
    if not (0 <= k < dim):
        raise ValueError(f"Eigenstate index k={k} out of range for n={n} qubits.")

    if j == float('inf') or np.isinf(j):
        # Computational basis state |k>
        state = np.zeros(dim, dtype=complex)
        state[k] = 1.0
        return state
    else:
        # Column k of U(j)
        u_mat = mub_unitary(n, int(j))
        return u_mat[:, k]


def sample_and_prepare_mub_state(
    weights: Dict[Union[int, float], float],
    n: int,
    wires: Optional[Sequence] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Union[int, float], int]:
    """
    Convenience function that samples (j, k) from the weighted distribution
    and immediately applies prepare_mub_state(n, j, k, wires=wires).

    Returns
    -------
    (j, k)
        The sampled basis and eigenstate index.
    """
    j, k = sample_mub_basis_and_state(weights, n, rng=rng)
    prepare_mub_state(n, j, k, wires=wires)
    return j, k
