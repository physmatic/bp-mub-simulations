import sys
from pathlib import Path
import io
import itertools
import math
from typing import List, Optional

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pennylane as qml

# Ensure project modules can be imported
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR.parent.parent))

try:
    from .mub_circuit import QUBIT_NUM, mub_circuit, mub_unitary
except ImportError:
    from mub_circuit import QUBIT_NUM, mub_circuit, mub_unitary



def validate_orthogonality(base: List[np.ndarray]) -> bool:
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            dot_prod = np.vdot(base[i], base[j])
            if not np.isclose(dot_prod, 0):
                return False
    return True


def validate_mubs(base1: List[np.ndarray], base2: List[np.ndarray]) -> bool:
    assert len(base1) == len(base2)
    dim = len(base1)
    for k in range(dim):
        for l in range(dim):
            dot_prod = np.vdot(base1[k], base2[l])
            if not np.isclose(abs(dot_prod) ** 2, 1 / dim):
                return False
    return True


def draw_circuits_grid(circuit_indices: List[Optional[int]], n: int = QUBIT_NUM) -> None:
    count = len(circuit_indices)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

    for ax in axs[count:]:
        ax.axis("off")

    for i, j in enumerate(circuit_indices):
        buf = io.BytesIO()
        if j is None:
            def empty_circuit():
                for q in range(n):
                    qml.Identity(wires=q)
            fig_circuit, _ = qml.draw_mpl(empty_circuit)()
            title = "Standard basis"
        else:
            fig_circuit, _ = qml.draw_mpl(mub_circuit)(n, j)
            title = f"MUB j={j}"

        fig_circuit.savefig(buf, format='pdf', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)

        axs[i].imshow(img)
        axs[i].axis("off")
        axs[i].set_title(title)
        plt.close(fig_circuit)

    plt.tight_layout()
    plt.show()


def generate_mubs(n: int = QUBIT_NUM, run_simultion: bool = False, plot: bool = True) -> List[List[np.ndarray]]:
    if isinstance(n, bool):
        run_simultion = n
        n = QUBIT_NUM

    dim = 2 ** n
    standard_basis = [np.eye(dim)[:, i] for i in range(dim)]
    mubs = [standard_basis]
    circuits: List[Optional[int]] = [None]

    for j in range(dim):
        circuits.append(j)

        # Unitary matrix for basis j
        u_j = mub_unitary(n, j)
        # Transformed basis states are the columns of U_j
        new_base = [u_j[:, i] for i in range(dim)]

        # Validate result
        assert validate_orthogonality(new_base), f"Basis j={j} is not orthogonal"
        for base in mubs:
            assert validate_mubs(base, new_base), f"Basis j={j} failed MUB test against previous basis"

        mubs.append(new_base)

        if run_simultion:
            dev_sim = qml.device("default.qubit", wires=n, shots=1024)

            @qml.qnode(dev_sim)
            def sim_node():
                mub_circuit(n, j)
                return qml.counts()

            counts = sim_node()
            plt.figure(figsize=(6, 4))
            plt.bar(list(counts.keys()), list(counts.values()))
            plt.title(f"Counts for MUB j={j}")
            plt.xlabel("State")
            plt.ylabel("Counts")
            plt.show()

    if plot:
        draw_circuits_grid(circuits, n)

    return mubs


if __name__ == '__main__':
    generate_mubs(plot=False)

