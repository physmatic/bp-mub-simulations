"""
Paulis-MUBs correspondence test for 2 qubits.

Evaluates the exact correspondence between the 5 MUB bases (Basis 0..4)
and the 5 maximal commuting Pauli stabilizer sets (C_0, C_1, C_2, C_3, C_4):

  C_0 = { Z (x) I, I (x) Z, Z (x) Z }  <--->  Basis 0 (Standard Basis)
  C_1 = { X (x) I, I (x) X, X (x) X }  <--->  Basis 1 (MUB j=0)
  C_2 = { Y (x) I, I (x) Y, Y (x) Y }  <--->  Basis 2 (MUB j=1)
  C_3 = { X (x) Z, Z (x) Y, Y (x) X }  <--->  Basis 3 (MUB j=2)
  C_4 = { Y (x) Z, Z (x) X, X (x) Y }  <--->  Basis 4 (MUB j=3)
"""

import sys
from pathlib import Path
import itertools
from typing import Dict, List
import numpy as np

# Ensure project modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "MUB_Circuits_generator"))

try:
    from src.MUB_Circuits_generator.mub_circuit import mub_circuit
except ImportError:
    from mub_circuit import mub_circuit

from qiskit.quantum_info import Statevector, Pauli


# -----------------------------------------------------------------------------
# 1. Define the 5 Commuting Pauli Sets in the Specified Ordering C_0 .. C_4
# -----------------------------------------------------------------------------
# Standard tensor convention: A (x) B where A acts on qubit 0 and B acts on qubit 1.
# In Qiskit's Pauli string (right-to-left endianness), Pauli(B + A) has:
# qubit 0 = A, qubit 1 = B.
PAULI_SETS = {
    "C_0": [
        ("Z (x) I", "IZ"),
        ("I (x) Z", "ZI"),
        ("Z (x) Z", "ZZ"),
    ],
    "C_1": [
        ("X (x) I", "IX"),
        ("I (x) X", "XI"),
        ("X (x) X", "XX"),
    ],
    "C_2": [
        ("Y (x) I", "IY"),
        ("I (x) Y", "YI"),
        ("Y (x) Y", "YY"),
    ],
    "C_3": [
        ("X (x) Z", "ZX"),
        ("Z (x) Y", "YZ"),
        ("Y (x) X", "XY"),
    ],
    "C_4": [
        ("Y (x) Z", "ZY"),
        ("Z (x) X", "XZ"),
        ("X (x) Y", "YX"),
    ],
}


# -----------------------------------------------------------------------------
# 2. Generate 2-Qubit MUBs (Basis 0 to Basis 4)
# -----------------------------------------------------------------------------
def generate_2qubit_mubs() -> Dict[str, List[Statevector]]:
    """
    Generates the 5 MUBs for 2 qubits (dimension d = 4):
    - Basis 0: Standard computational basis
    - Basis 1..4: Transformed via mub_circuit(n=2, j) for j in {0, 1, 2, 3}
    """
    n = 2
    standard_labels = [''.join(bits) for bits in itertools.product('01', repeat=n)]
    standard_basis = [Statevector.from_label(lbl) for lbl in standard_labels]

    mub_bases = {
        "Basis 0 (Standard Basis)": standard_basis
    }

    for j in range(2 ** n):
        qc = mub_circuit(n, j)
        transformed_basis = [sv.evolve(qc) for sv in standard_basis]
        mub_bases[f"Basis {j+1} (MUB j={j})"] = transformed_basis

    return mub_bases


# -----------------------------------------------------------------------------
# 3. Evaluate Pauli Expectation Values <state | P | state>
# -----------------------------------------------------------------------------
def evaluate_expectations(mub_bases: Dict[str, List[Statevector]]) -> None:
    print("=" * 80)
    print(" 2-QUBIT MUBs vs. ORDERED PAULI STABILIZER SETS (C_0 ... C_4)")
    print("=" * 80)

    basis_to_stabilizer = {}

    for b_idx, (basis_name, states) in enumerate(mub_bases.items()):
        print(f"\n{'#' * 80}")
        print(f" >>> {basis_name.upper()} (4 States) <<<")
        print(f"{'#' * 80}")

        matched_set = None

        for set_name, paulis in PAULI_SETS.items():
            set_desc = ", ".join([p[0] for p in paulis])
            print(f"\n--- {set_name} = {{ {set_desc} }} ---")
            header = f"{'State':<12}" + "".join([f"{label:>18}" for label, _ in paulis])
            print(header)
            print("-" * len(header))

            set_expectations = []
            for s_idx, state in enumerate(states):
                row = f"|psi_{s_idx}>      "
                state_evs = []
                for label, qiskit_str in paulis:
                    p_op = Pauli(qiskit_str)
                    ev = np.round(state.expectation_value(p_op).real, 4)
                    state_evs.append(ev)
                    row += f"{ev:>18.2f}"
                print(row)
                set_expectations.append(state_evs)

            # Check if this Pauli set stabilizes this basis (all expectations are ±1)
            arr = np.array(set_expectations)
            if np.all(np.isclose(np.abs(arr), 1.0)):
                matched_set = set_name

        if matched_set:
            basis_to_stabilizer[basis_name] = matched_set

    # -------------------------------------------------------------------------
    # 4. Summary Table of Exact 1-to-1 Correspondence
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(" EXACT 1-TO-1 CORRESPONDENCE SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'MUB Basis':<30} <---> {'Stabilizer Set':<10} {'Operators'}")
    print("-" * 80)
    for basis_name, set_name in basis_to_stabilizer.items():
        ops_str = ", ".join([p[0] for p in PAULI_SETS[set_name]])
        print(f"{basis_name:<30} <---> {set_name:<10} {{ {ops_str} }}")
    print("=" * 80)


if __name__ == "__main__":
    bases = generate_2qubit_mubs()
    evaluate_expectations(bases)
