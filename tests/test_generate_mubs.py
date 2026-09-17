"""Deterministic evaluation of Mutually Unbiased Bases (MUBs) for n = 3 qubits (d = 8) and n = 4 qubits (d = 16).

No shots or sampling are used: state vectors are evaluated deterministically
via exact linear algebra.

For all pairs (j, l) of bases and states (i, k):
  P = |<psi_i^j | psi_k^l>|^2
  - If j == l and i == k:  P == 1
  - If j == l and i != k:  P == 0
  - If j != l:             P == 1/d
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "MUB_Circuits_generator"))

from src.MUB_Circuits_generator.generate_mubs import generate_mubs


def test_3qubit_mubs_deterministic_overlaps():
    n_qubits = 3
    dim = 2 ** n_qubits  # d = 8
    num_bases = dim + 1   # 9 bases
    tol = 1e-12
    expected_inter_p = 1.0 / dim  # 1/8 = 0.125

    print("=" * 88)
    print(f"DETERMINISTIC EVALUATION OF ALL PAIRWISE OVERLAPS: |<psi_i^j | psi_k^l>|^2")
    print(f"n = {n_qubits} qubits, dimension d = {dim}")
    print(f"(No shots/sampling: exact state vectors evaluated via inner products)")
    print("=" * 88)
    print(f"Basis index j, l in {{0, 1, ..., {num_bases - 1}}} (9 bases total)")
    print(f"State index i, k in {{0, 1, ..., {dim - 1}}} (8 states per basis)")
    print()
    print("Theoretical requirements:")
    print(f"  1. If j == l and i == k : |<psi_i^j | psi_k^l>|^2 = 1.000000 (Normalization)")
    print(f"  2. If j == l and i != k : |<psi_i^j | psi_k^l>|^2 = 0.000000 (Orthogonality within basis)")
    print(f"  3. If j != l            : |<psi_i^j | psi_k^l>|^2 = 1/8 = {expected_inter_p:.6f} (Mutual unbiasedness)")
    print("=" * 88)

    # 1. Deterministic generation of all state vectors
    print(f"\n[1] Generating exact state vectors for {num_bases} bases (d = {dim})...")
    bases = generate_mubs(n=n_qubits, plot=False)
    assert len(bases) == num_bases, f"Expected {num_bases} bases, got {len(bases)}"
    for b_idx, base in enumerate(bases):
        assert len(base) == dim, f"Basis {b_idx} has {len(base)} states, expected {dim}"
    print(f"    Done. 9 bases x 8 states = {num_bases * dim} state vectors total.")

    basis_names = ["Standard (inf)"] + [f"Clifford j={j}" for j in range(dim)]

    # 2. Flatten all 72 states into matrix of shape (8, 72)
    all_states = np.column_stack([np.column_stack(base) for base in bases])
    gram_matrix = all_states.conj().T @ all_states  # shape: (72, 72)
    overlap_sq = np.abs(gram_matrix) ** 2            # shape: (72, 72)

    total_pairs = (num_bases * dim) ** 2  # 72 * 72 = 5,184

    # 3. Categorize all 5,184 pairs into the 3 cases
    count_diag = 0       # j == l, i == k
    count_intra_off = 0  # j == l, i != k
    count_inter = 0      # j != l

    max_dev_diag = 0.0
    max_dev_intra_off = 0.0
    max_dev_inter = 0.0

    min_val_inter = float("inf")
    max_val_inter = 0.0

    for j in range(num_bases):
        for l in range(num_bases):
            block = overlap_sq[j * dim : (j + 1) * dim, l * dim : (l + 1) * dim]

            for i in range(dim):
                for k in range(dim):
                    val = block[i, k]

                    if j == l and i == k:
                        count_diag += 1
                        dev = abs(val - 1.0)
                        if dev > max_dev_diag:
                            max_dev_diag = dev
                        assert dev < tol, f"Failed norm: j={j}, i={i}, val={val}"

                    elif j == l and i != k:
                        count_intra_off += 1
                        dev = abs(val - 0.0)
                        if dev > max_dev_intra_off:
                            max_dev_intra_off = dev
                        assert dev < tol, f"Failed orthogonality: j={j}, i={i}, k={k}, val={val}"

                    else:  # j != l
                        count_inter += 1
                        dev = abs(val - expected_inter_p)
                        if val < min_val_inter:
                            min_val_inter = val
                        if val > max_val_inter:
                            max_val_inter = val
                        if dev > max_dev_inter:
                            max_dev_inter = dev
                        assert dev < tol, f"Failed MUB unbiasedness: j={j}, l={l}, i={i}, k={k}, val={val}"

    # 4. Summary Across All Pairs
    print("\n[2] Comprehensive Summary Across ALL 5,184 Evaluated Pairs:")
    print("-" * 88)
    print(f"{'Condition':<30} | {'Count':<10} | {'Expected':<12} | {'Observed Range':<22} | {'Max Deviation'}")
    print("-" * 88)
    print(f"{'Same basis, same state (j=l, i=k)':<30} | {count_diag:<10} | {'1.000000':<12} | {'[1.000000, 1.000000]':<22} | {max_dev_diag:.2e}")
    print(f"{'Same basis, diff state (j=l, i!=k)':<30} | {count_intra_off:<10} | {'0.000000':<12} | {'[0.000000, 0.000000]':<22} | {max_dev_intra_off:.2e}")
    print(f"{'Diff bases (j!=l, any i,k)':<30} | {count_inter:<10} | {'1/8=0.125000':<12} | {f'[{min_val_inter:.6f}, {max_val_inter:.6f}]':<22} | {max_dev_inter:.2e}")
    print("-" * 88)
    print(f"Total pairs evaluated: {count_diag + count_intra_off + count_inter:,} (exactly 72 x 72).")

    # 5. Print ALL 9 Intra-Basis Blocks (j = l)
    print("\n[3] Verification of ALL 9 Intra-Basis Blocks (j = l, 0 <= j <= 8):")
    print("-" * 88)
    print(f"{'Basis Index j':<14} | {'Basis Name':<18} | {'Diagonal (min, max)':<22} | {'Max Off-Diag Overlap'}")
    print("-" * 88)
    for j in range(num_bases):
        block_jj = overlap_sq[j * dim : (j + 1) * dim, j * dim : (j + 1) * dim]
        diag_vals = np.diag(block_jj)
        off_diag = block_jj - np.diag(diag_vals)
        max_off = np.max(np.abs(off_diag))
        diag_range_str = f"[{np.min(diag_vals):.6f}, {np.max(diag_vals):.6f}]"
        print(f"{j:<14} | {basis_names[j]:<18} | {diag_range_str:<22} | {max_off:.2e}")
    print("-" * 88)
    print("✓ All 9 bases are strictly orthonormal (diagonal == 1.0, off-diagonal == 0.0).")

    # 6. Print ALL 36 Distinct Inter-Basis Pairs (j < l)
    num_pairs = (num_bases * (num_bases - 1)) // 2  # 36
    print(f"\n[4] Complete Results for ALL {num_pairs} Distinct Inter-Basis Pairs (j < l):")
    print(f"    (Each basis pair contains exactly 8 x 8 = 64 state pairs |<psi_i^j | psi_k^l>|^2)")
    print("-" * 88)
    print(f"{'#':<3} | {'Basis Pair (j, l)':<40} | {'Min P':<10} | {'Max P':<10} | {'Max Dev from 1/8':<16}")
    print("-" * 88)
    pair_idx = 1
    for j in range(num_bases):
        for l in range(j + 1, num_bases):
            block_jl = overlap_sq[j * dim : (j + 1) * dim, l * dim : (l + 1) * dim]
            min_p = np.min(block_jl)
            max_p = np.max(block_jl)
            max_dev = np.max(np.abs(block_jl - expected_inter_p))
            pair_name = f"({j}, {l}) [{basis_names[j]} vs {basis_names[l]}]"
            print(f"{pair_idx:<3} | {pair_name:<40} | {min_p:<10.6f} | {max_p:<10.6f} | {max_dev:<16.2e}")
            pair_idx += 1
    print("-" * 88)
    print(f"✓ ALL {num_pairs} basis pairs evaluated: EVERY SINGLE state overlap is strictly 1/8 = 0.125000.")

    # 7. Print Full 8x8 Overlap Matrix for Representative Basis Pairs
    print("\n[5] Full 8x8 Overlap Matrix |<psi_i^j | psi_k^l>|^2 for Basis Pair (j=0, l=1):")
    print("    Standard Basis vs Clifford Basis j=0 (every entry must be 0.125000):")
    block_0_1 = overlap_sq[0:dim, dim:2*dim]
    header = "       " + "  ".join(f"k={k}" for k in range(dim))
    print(header)
    for r in range(dim):
        row_str = " ".join(f"{block_0_1[r, c]:.6f}" for c in range(dim))
        print(f"  i={r} | {row_str}")

    print("\n[6] Full 8x8 Overlap Matrix |<psi_i^j | psi_k^l>|^2 for Basis Pair (j=1, l=2):")
    print("    Clifford Basis j=0 vs Clifford Basis j=1 (every entry must be 0.125000):")
    block_1_2 = overlap_sq[dim:2*dim, 2*dim:3*dim]
    print(header)
    for r in range(dim):
        row_str = " ".join(f"{block_1_2[r, c]:.6f}" for c in range(dim))
        print(f"  i={r} | {row_str}")

    print("\n" + "=" * 88)
    print("ALL RESULTS CONFIRMED FOR n = 3 QUBITS (d = 8):")
    print(f"  ✓ 9 mutually unbiased bases, 8 states each (72 states total)")
    print(f"  ✓ Intra-basis: 1.0 for i == k, 0.0 for i != k")
    print(f"  ✓ Inter-basis: strictly 1/8 = 0.125000 for all 36 basis pairs and all 2,304 state pairs")
    print("=" * 88)


def test_4qubit_mubs_deterministic_overlaps():
    n_qubits = 4
    dim = 2 ** n_qubits  # d = 16
    num_bases = dim + 1   # 17 bases
    tol = 1e-12

    bases = generate_mubs(n=n_qubits, plot=False)
    all_states = np.column_stack([np.column_stack(base) for base in bases])
    gram_matrix = all_states.conj().T @ all_states
    overlap_sq = np.abs(gram_matrix) ** 2

    for j in range(num_bases):
        for l in range(num_bases):
            block = overlap_sq[j * dim : (j + 1) * dim, l * dim : (l + 1) * dim]
            for i in range(dim):
                for k in range(dim):
                    val = block[i, k]
                    if j == l and i == k:
                        assert abs(val - 1.0) < tol
                    elif j == l and i != k:
                        assert abs(val - 0.0) < tol
                    else:
                        assert abs(val - 1.0 / 16.0) < tol


if __name__ == "__main__":
    test_3qubit_mubs_deterministic_overlaps()
