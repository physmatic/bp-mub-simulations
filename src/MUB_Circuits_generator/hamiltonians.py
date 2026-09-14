"""
Hamiltonian generators for quantum simulation and benchmark sweeps.
"""

from typing import List, Sequence, Optional
import pennylane as qml


def build_tfim_hamiltonian(
    n: int,
    J: float = 1.0,
    h: float = 1.0,
    wires: Optional[Sequence] = None
) -> qml.Hamiltonian:
    """
    Constructs the 1D Transverse-Field Ising Model (TFIM) Hamiltonian
    with open-chain boundary conditions:

        H = -J * sum_{i=0}^{n-2} Z_i Z_{i+1}  -  h * sum_{i=0}^{n-1} X_i

    Parameters
    ----------
    n : int
        Number of qubits (spins in the chain). Must be >= 1.
    J : float, default=1.0
        Coupling constant for nearest-neighbor ZZ interactions.
    h : float, default=1.0
        Strength of the transverse X magnetic field.
    wires : Sequence, optional
        Custom wire identifiers of length n. If None, defaults to list(range(n)).

    Returns
    -------
    qml.Hamiltonian
        PennyLane Hamiltonian with open boundary conditions.
        For n >= 2, contains (n - 1) ZZ terms and n X terms (total 2n - 1 terms).
    """
    if n < 1:
        raise ValueError(f"Number of qubits n must be at least 1, got {n}")

    if wires is None:
        wire_list = list(range(n))
    else:
        wire_list = list(wires)
        if len(wire_list) != n:
            raise ValueError(f"Expected {n} wires, got {len(wire_list)}")

    coeffs: List[float] = []
    ops: List[qml.operation.Operator] = []

    # 1. Nearest-neighbor ZZ interactions (open chain: 0 to n-2)
    if n > 1 and J != 0.0:
        for i in range(n - 1):
            coeffs.append(-float(J))
            ops.append(qml.PauliZ(wire_list[i]) @ qml.PauliZ(wire_list[i + 1]))

    # 2. Transverse magnetic field X_i (all spins: 0 to n-1)
    if h != 0.0:
        for i in range(n):
            coeffs.append(-float(h))
            ops.append(qml.PauliX(wire_list[i]))

    # Return empty Hamiltonian if all coefficients were zero
    if not ops:
        return qml.Hamiltonian([], [])

    return qml.Hamiltonian(coeffs, ops)
