"""
Hardware-Efficient Ansatz (HEA) for PennyLane.

Consists of alternating layers of single-qubit Ry rotations on each qubit,
followed by a linear entangling chain of CZ gates between adjacent qubits (i, i+1).
A final layer of Ry rotations is applied at the end.

Structure:
  - For l = 0 to layers - 1:
      - Ry(theta_{l, i}) on wire i for all i in [0, n-1]
      - CZ on wires [i, i+1] for all i in [0, n-2]
  - Final layer:
      - Ry(theta_{layers, i}) on wire i for all i in [0, n-1]

Total rotation layers: layers + 1
Total entangling layers: layers
Total parameters: (layers + 1) * n
"""

from typing import Optional, Sequence, Union
import numpy as np
import pennylane as qml


def get_num_hea_params(n: int, layers: int) -> int:
    """
    Returns the total number of variational parameters for an n-qubit HEA
    with the given number of layers.

    Parameters
    ----------
    n : int
        Number of qubits.
    layers : int
        Number of entangling layers.

    Returns
    -------
    int
        Total number of parameters: (layers + 1) * n.
    """
    if n < 1:
        raise ValueError(f"Number of qubits must be >= 1, got {n}")
    if layers < 0:
        raise ValueError(f"Number of layers must be >= 0, got {layers}")
    return (layers + 1) * n


def hardware_efficient_ansatz(
    params: Union[np.ndarray, Sequence[float]],
    wires: Optional[Sequence] = None,
    layers: Optional[int] = None,
) -> None:
    """
    Applies the Hardware-Efficient Ansatz on the specified quantum register.

    Parameters
    ----------
    params : array-like
        Variational parameters, either flat of length (layers + 1) * n,
        or 2D of shape (layers + 1, n).
    wires : Sequence, optional
        Ordered wire labels. If None, inferred from params shape or defaults to range(n).
    layers : int, optional
        Number of entangling layers. If None, inferred from len(params) and len(wires).
    """
    flat_params = np.asarray(params).flatten()

    if wires is None:
        if layers is not None:
            n = len(flat_params) // (layers + 1)
        else:
            raise ValueError("Must provide either 'wires' or 'layers' to infer ansatz dimensions.")
        wire_list = list(range(n))
    else:
        wire_list = list(wires)
        n = len(wire_list)

    if layers is None:
        expected_params_per_layer = n
        if len(flat_params) % expected_params_per_layer != 0:
            raise ValueError(
                f"Parameter length {len(flat_params)} is not divisible by qubit count {n}."
            )
        layers = (len(flat_params) // expected_params_per_layer) - 1

    expected_total = get_num_hea_params(n, layers)
    if len(flat_params) != expected_total:
        raise ValueError(
            f"Expected {expected_total} parameters for n={n}, layers={layers}, got {len(flat_params)}."
        )

    # Reshape parameters to (layers + 1, n)
    reshaped_params = flat_params.reshape((layers + 1, n))

    # Apply layers
    for l in range(layers):
        # Single-qubit Ry rotations
        for i in range(n):
            qml.RY(reshaped_params[l, i], wires=wire_list[i])
        # Linear entangling chain of CZ gates
        for i in range(n - 1):
            qml.CZ(wires=[wire_list[i], wire_list[i + 1]])

    # Final single-qubit Ry rotation layer
    for i in range(n):
        qml.RY(reshaped_params[layers, i], wires=wire_list[i])
