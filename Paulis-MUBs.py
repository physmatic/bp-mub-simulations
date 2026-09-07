"""
Paulis-MUBs correspondence test for n qubits.

Evaluates the exact correspondence between the (2^n + 1) MUB bases
and the (2^n + 1) maximal commuting Pauli stabilizer sets.
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
    from src.MUB_Circuits_generator.pauli_generator import generate_pauli_sets
except ImportError:
    from mub_circuit import mub_circuit
    from pauli_generator import generate_pauli_sets

from qiskit.quantum_info import Statevector, Pauli


# -----------------------------------------------------------------------------
# 1. Generate n-Qubit MUBs (Basis 0 to Basis 2^n)
# -----------------------------------------------------------------------------
def generate_nqubit_mubs(n: int) -> Dict[str, List[Statevector]]:
    """
    Generates the (2^n + 1) MUBs for n qubits.
    - Basis 0: Standard computational basis
    - Basis 1..2^n: Transformed via mub_circuit(n, j)
    """
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
# 2. Evaluate Pauli Expectation Values <state | P | state>
# -----------------------------------------------------------------------------
def evaluate_expectations(n: int, mub_bases: Dict[str, List[Statevector]], pauli_sets: List[List[str]]) -> None:
    print("=" * 80)
    print(f" {n}-QUBIT MUBs vs. DYNAMIC PAULI STABILIZER SETS (C_0 ... C_{2**n})")
    print("=" * 80)

    basis_to_stabilizer = {}

    for b_idx, (basis_name, states) in enumerate(mub_bases.items()):
        matched_set = None

        for c_idx, pauli_strings in enumerate(pauli_sets):
            set_name = f"C_{c_idx}"
            
            set_expectations = []
            for s_idx, state in enumerate(states):
                state_evs = []
                for qiskit_str in pauli_strings:
                    p_op = Pauli(qiskit_str)
                    ev = np.round(state.expectation_value(p_op).real, 4)
                    state_evs.append(ev)
                set_expectations.append(state_evs)

            # Check if this Pauli set stabilizes this basis (all expectations are ±1)
            arr = np.array(set_expectations)
            if np.all(np.isclose(np.abs(arr), 1.0)):
                matched_set = set_name

        if matched_set:
            basis_to_stabilizer[basis_name] = matched_set
            print(f"[OK] {basis_name:<30} <---> {matched_set}")
        else:
            print(f"[FAIL] {basis_name:<30} <---> NO MATCHING SET FOUND")

    # -------------------------------------------------------------------------
    # 3. Summary Table of Exact 1-to-1 Correspondence
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f" EXACT 1-TO-1 CORRESPONDENCE SUMMARY FOR {n} QUBITS")
    print(f"{'=' * 80}")
    print(f"{'MUB Basis':<30} <---> {'Stabilizer Set':<15}")
    print("-" * 80)
    for basis_name, set_name in basis_to_stabilizer.items():
        print(f"{basis_name:<30} <---> {set_name:<15}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test MUB to Pauli Stabilizer Correspondence.")
    parser.add_argument("-n", "--qubits", type=int, default=3, help="Number of qubits (default: 3)")
    args = parser.parse_args()

    n = args.qubits
    print(f"Generating MUBs and Pauli sets for n={n} qubits...")
    
    pauli_sets = generate_pauli_sets(n)
    bases = generate_nqubit_mubs(n)
    
    evaluate_expectations(n, bases, pauli_sets)
