"""
Hardware-Efficient Ansatz (HEA) for PennyLane.

Alternating layers of single-qubit rotations on each qubit, followed by a linear
entangling chain of CZ gates between adjacent qubits (i, i+1). A final rotation
layer is applied at the end.

Structure:
  - For l = 0 to layers - 1:
      - rotation block on wire i for all i in [0, n-1]
      - CZ on wires [i, i+1] for all i in [0, n-2]
  - Final layer:
      - rotation block on wire i for all i in [0, n-1]

Rotation block ("gate_set"):
  - "ry"   : Ry(theta)                      -> 1 parameter per qubit per layer
  - "ryrz" : Ry(theta) followed by Rz(phi)  -> 2 parameters per qubit per layer

Why "ryrz" is the default for experiments
-----------------------------------------
With "ry" every layer applies the *same* CZ chain C, and C^2 = I. The circuit can
then only generate the Lie algebra spanned by {Y_i, C Y_i C = Z_{i-1} Y_i Z_{i+1}},
whose dimension is n(n+1) (checked numerically for n = 2..9). Such a polynomial
dynamical Lie algebra means the ansatz never approaches a 2-design at any depth:
gradient variances decay only polynomially (~1/n), there is no barren plateau, and
the circuit is classically simulable. Adding Rz makes the algebra all of su(2^n).
"ry" is kept only to reproduce the legacy results in outputs/old/.

Total parameters: (layers + 1) * n * params_per_qubit(gate_set)
"""

from typing import Optional, Sequence, Union
import numpy as np
import pennylane as qml


GATE_SETS = {"ry": 1, "ryrz": 2}


def params_per_qubit(gate_set: str) -> int:
    """Number of parameters in one rotation block on one qubit."""
    try:
        return GATE_SETS[gate_set]
    except KeyError:
        raise ValueError(f"Unknown gate_set '{gate_set}'. Choose one of {list(GATE_SETS)}.")


def get_num_hea_params(n: int, layers: int, gate_set: str = "ry") -> int:
    """
    Returns the total number of variational parameters for an n-qubit HEA
    with the given number of entangling layers: (layers + 1) * n * params_per_qubit.
    """
    if n < 1:
        raise ValueError(f"Number of qubits must be >= 1, got {n}")
    if layers < 0:
        raise ValueError(f"Number of layers must be >= 0, got {layers}")
    return (layers + 1) * n * params_per_qubit(gate_set)


def hardware_efficient_ansatz(
    params: Union[np.ndarray, Sequence[float]],
    wires: Optional[Sequence] = None,
    layers: Optional[int] = None,
    gate_set: str = "ry",
) -> None:
    """
    Applies the Hardware-Efficient Ansatz on the specified quantum register.

    Parameters
    ----------
    params : array-like
        Flat variational parameters of length (layers + 1) * n * params_per_qubit.
        Ordering: layer-major, then qubit, then rotation within the block
        (for "ryrz": [Ry_q0, Rz_q0, Ry_q1, Rz_q1, ...] for each layer).
    wires : Sequence, optional
        Ordered wire labels. If None, inferred from len(params) and layers.
    layers : int, optional
        Number of entangling layers. If None, inferred from len(params) and len(wires).
    gate_set : str, default="ry"
        Rotation block: "ry" or "ryrz" (see module docstring).
    """
    ppq = params_per_qubit(gate_set)
    flat_params = np.asarray(params).flatten()

    if wires is None:
        if layers is None:
            raise ValueError("Must provide either 'wires' or 'layers' to infer ansatz dimensions.")
        n = len(flat_params) // ((layers + 1) * ppq)
        wire_list = list(range(n))
    else:
        wire_list = list(wires)
        n = len(wire_list)

    if layers is None:
        if len(flat_params) % (n * ppq) != 0:
            raise ValueError(
                f"Parameter length {len(flat_params)} is not divisible by n * params_per_qubit = {n * ppq}."
            )
        layers = (len(flat_params) // (n * ppq)) - 1

    expected_total = get_num_hea_params(n, layers, gate_set)
    if len(flat_params) != expected_total:
        raise ValueError(
            f"Expected {expected_total} parameters for n={n}, layers={layers}, "
            f"gate_set={gate_set}, got {len(flat_params)}."
        )

    p = flat_params.reshape((layers + 1, n, ppq))

    def rotation_layer(l: int) -> None:
        for i in range(n):
            qml.RY(p[l, i, 0], wires=wire_list[i])
            if gate_set == "ryrz":
                qml.RZ(p[l, i, 1], wires=wire_list[i])

    for l in range(layers):
        rotation_layer(l)
        for i in range(n - 1):
            qml.CZ(wires=[wire_list[i], wire_list[i + 1]])

    rotation_layer(layers)
