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


def build_xy_dm_hamiltonian(
    n: int,
    Jx: float = 1.0,
    Jy: float = 0.5,
    D: float = 0.8,
    h: float = 1.0,
    wires: Optional[Sequence] = None
) -> qml.Hamiltonian:
    r"""
    Constructs the 1D Transverse-Field XY Model with Dzyaloshinskii-Moriya (DM) interaction:

        H = -\sum_{i=0}^{n-2} \left( J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + D(X_i Y_{i+1} - Y_i X_{i+1}) \right)
            - h \sum_{i=0}^{n-1} Z_i

    Parameters
    ----------
    n : int
        Number of qubits (spins in the chain). Must be >= 1.
    Jx : float, default=1.0
        Coupling constant for nearest-neighbor XX interactions.
    Jy : float, default=0.5
        Coupling constant for nearest-neighbor YY interactions.
    D : float, default=0.8
        Strength of the DM antisymmetric cross-term interaction (X_i Y_{i+1} - Y_i X_{i+1}).
    h : float, default=1.0
        Strength of the transverse Z magnetic field.
    wires : Sequence, optional
        Custom wire identifiers of length n. If None, defaults to list(range(n)).

    Returns
    -------
    qml.Hamiltonian
        PennyLane Hamiltonian with open boundary conditions.
        Contains up to 4*(n - 1) nearest-neighbor terms and n transverse Z terms (total 5n - 4 terms).
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

    # 1. Nearest-neighbor interactions (i = 0 to n-2)
    if n > 1:
        for i in range(n - 1):
            w1, w2 = wire_list[i], wire_list[i + 1]

            if Jx != 0.0:
                coeffs.append(-float(Jx))
                ops.append(qml.PauliX(w1) @ qml.PauliX(w2))

            if Jy != 0.0:
                coeffs.append(-float(Jy))
                ops.append(qml.PauliY(w1) @ qml.PauliY(w2))

            if D != 0.0:
                # - D * X_i Y_{i+1}
                coeffs.append(-float(D))
                ops.append(qml.PauliX(w1) @ qml.PauliY(w2))

                # + D * Y_i X_{i+1}
                coeffs.append(float(D))
                ops.append(qml.PauliY(w1) @ qml.PauliX(w2))

    # 2. Transverse magnetic field Z_i (all spins: 0 to n-1)
    if h != 0.0:
        for i in range(n):
            coeffs.append(-float(h))
            ops.append(qml.PauliZ(wire_list[i]))

    if not ops:
        return qml.Hamiltonian([], [])

    return qml.Hamiltonian(coeffs, ops)

