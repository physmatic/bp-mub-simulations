import sys
from pathlib import Path
import time
import random
import itertools
from typing import Optional, Union, Tuple
import numpy as np
import pennylane as qml

# Ensure project modules can be imported
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .mub_circuit import mub_circuit, mub_unitary, compute_field_trace, _gf2_multiply
    from .consts import IRREDUCIBLE_POLYS
except ImportError:
    from mub_circuit import mub_circuit, mub_unitary, compute_field_trace, _gf2_multiply
    from consts import IRREDUCIBLE_POLYS




# Single-qubit mapping: char -> (x, z)
_PAULI_MAP = {
    'I': (0, 0),
    'X': (1, 0),
    'Y': (1, 1),
    'Z': (0, 1),
}

def pauli_to_xz(pauli, qiskit_string_order: bool = False, n: Optional[int] = None):
    """
    Map an n-qubit Pauli operator to binary vectors x and z such that
    P_k = i^(x_k * z_k) * X^(x_k) * Z^(z_k).

    Parameters
    ----------
    pauli : str, qml.operation.Operator, or legacy Pauli object
        Pauli string (e.g., 'IXYZ') or PennyLane operator.
    qiskit_string_order : bool, default=False
        Only relevant if `pauli` is passed as a raw string.
        If False (PennyLane / standard default): index 0 corresponds to wire 0 (left-to-right).
        If True (legacy Qiskit): rightmost char corresponds to qubit 0.
    n : int, optional
        Number of qubits/wires (inferred if None).

    Returns
    -------
    x : np.ndarray of shape (n,), dtype=np.uint8
    z : np.ndarray of shape (n,), dtype=np.uint8
    """
    # 1. Native Qiskit Pauli object (legacy support)
    if hasattr(pauli, 'x') and hasattr(pauli, 'z'):
        return np.asarray(pauli.x, dtype=np.uint8), np.asarray(pauli.z, dtype=np.uint8)

    # 2. String input
    if isinstance(pauli, str):
        pauli_str = pauli.strip().upper()
        if qiskit_string_order:
            pauli_str = pauli_str[::-1]

        num_qubits = len(pauli_str)
        x = np.empty(num_qubits, dtype=np.uint8)
        z = np.empty(num_qubits, dtype=np.uint8)

        for idx, char in enumerate(pauli_str):
            if char not in _PAULI_MAP:
                raise ValueError(f"Invalid Pauli character '{char}'. Expected one of I, X, Y, Z.")
            x[idx], z[idx] = _PAULI_MAP[char]

        return x, z

    # 3. PennyLane Operator / PauliWord / PauliSentence
    if hasattr(pauli, "wires"):
        wires = list(pauli.wires)
        total_qubits = n if n is not None else (max(wires) + 1 if wires else 1)
        x = np.zeros(total_qubits, dtype=np.uint8)
        z = np.zeros(total_qubits, dtype=np.uint8)

        name_to_char = {
            "PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I",
            "I": "I", "X": "X", "Y": "Y", "Z": "Z"
        }

        if hasattr(pauli, "name") and pauli.name in name_to_char:
            wire = pauli.wires[0]
            x[wire], z[wire] = _PAULI_MAP[name_to_char[pauli.name]]
            return x, z

        if hasattr(pauli, "operands"):
            for op in pauli.operands:
                sub_x, sub_z = pauli_to_xz(op, n=total_qubits)
                x ^= sub_x
                z ^= sub_z
            return x, z

    raise TypeError(f"Unsupported Pauli type: {type(pauli)}")



def get_precomputed_traces(n: int, poly: int) -> np.ndarray:
    """
    Precomputes field trace for alpha^m for m in [0, 3*n - 3].
    Returns an array `traces` of length 3*n - 2.
    """
    max_power = 3 * n - 2
    traces = np.empty(max_power, dtype=np.uint8)
    curr = 1  # alpha^0 = 1
    alpha = 2  # alpha^1 = x
    for m in range(max_power):
        traces[m] = compute_field_trace(curr, poly, n)
        curr = _gf2_multiply(curr, alpha, poly, n)
    return traces


def build_M_tensors(n: int, poly: int) -> np.ndarray:
    """
    Constructs the 3D tensor M_tensor of shape (n, n, n), where:
        M_tensor[k, s, t] = M^{(k)}_{s, t} = Tr(alpha^(s + t + k))
    """
    traces = get_precomputed_traces(n, poly)
    M_tensor = np.empty((n, n, n), dtype=np.uint8)
    for k in range(n):
        for s in range(n):
            for t in range(n):
                M_tensor[k, s, t] = traces[s + t + k]
    return M_tensor


def solve_j(x: np.ndarray, z: np.ndarray, poly: int = None):
    """
    Solves for j such that:
        z = sum_{k=0}^{n-1} j_k (M^{(k)} x)  (mod 2)

    Returns
    -------
    j : int or float('inf')
        Integer in [0, 2^n - 1] representing the basis index,
        or float('inf') if x is the all-zero vector (computational basis).
    """
    x = np.asarray(x, dtype=np.uint8).flatten()
    z = np.asarray(z, dtype=np.uint8).flatten()
    n = len(x)

    # 1. Check for x == 0 -> computational basis (j = infinity)
    if not np.any(x):
        return float('inf')

    if poly is None:
        if n not in IRREDUCIBLE_POLYS:
            raise ValueError(f"No default irreducible polynomial defined for n={n}")
        poly = IRREDUCIBLE_POLYS[n]

    # 2. Build M tensors and form V = [v_0 | ... | v_{n-1}]
    M_tensor = build_M_tensors(n, poly)
    V = np.empty((n, n), dtype=np.uint8)
    for k in range(n):
        V[:, k] = (M_tensor[k] @ x) % 2

    # 3. Gaussian elimination over GF(2)
    aug = np.hstack([V.copy(), z[:, None]])
    pivot_row = 0

    for col in range(n):
        rows_with_one = np.where(aug[pivot_row:, col] == 1)[0]
        if len(rows_with_one) == 0:
            continue
        swap_idx = pivot_row + rows_with_one[0]
        aug[[pivot_row, swap_idx]] = aug[[swap_idx, pivot_row]]

        for r in range(n):
            if r != pivot_row and aug[r, col] == 1:
                aug[r] ^= aug[pivot_row]

        pivot_row += 1
        if pivot_row == n:
            break

    # Consistency check
    for r in range(pivot_row, n):
        if aug[r, n] != 0:
            raise ValueError("No solution exists for j over F_2 (inconsistent system).")

    # 4. Extract vector j_vec
    j_vec = np.zeros(n, dtype=np.uint8)
    for r in range(pivot_row - 1, -1, -1):
        col = np.where(aug[r, :n] == 1)[0][0]
        j_vec[col] = aug[r, n]

    # 5. Convert binary vector j_vec to integer j in {0, ..., 2^n - 1}
    # Little-endian convention: j_0 is 2^0, j_1 is 2^1, ...
    j_int = int(sum(int(bit) << k for k, bit in enumerate(j_vec)))

    return j_int


def p_to_j(pauli, n: Optional[int] = None) -> Union[int, float]:
    """
    Maps an n-qubit Pauli operator to its unique stabilizer basis index j in
    {0, ..., 2^n - 1} U {inf} (Clifford / math convention).

    Parameters
    ----------
    pauli : str or qml.operation.Operator
        Pauli string (e.g., 'IXYZ') or PennyLane operator.
    n : int, optional
        Number of qubits/wires (inferred from pauli if None).

    Returns
    -------
    int or float('inf')
        Integer j in {0, ..., 2^n - 1} corresponding to the Clifford parameter
        for mub_circuit(n, j), or float('inf') for the standard computational basis.
    """
    x, z = pauli_to_xz(pauli, n=n)
    if not np.any(x) and not np.any(z):
        raise ValueError("Identity operator I^n has no unique stabilizer basis.")

    return solve_j(x, z)



def generate_all_paulis(n: int):
    """
    Generator for all 4^n Pauli strings.
    Warning: only practical for n <= 7.
    """
    for p in itertools.product(["I", "X", "Y", "Z"], repeat=n):
        yield "".join(p)

def generate_random_pauli(n: int) -> str:
    """Generates a single random Pauli string of length n."""
    return "".join(random.choices(["I", "X", "Y", "Z"], k=n))

def run_pauli_solver(n: int, num_samples: int = 10, print_results: bool = True):
    """
    Runs the solver for a given n. If 4^n <= num_samples, processes all operators;
    otherwise, benchmarks on random samples.
    """
    if n not in IRREDUCIBLE_POLYS:
        raise ValueError(f"Irreducible polynomial not defined for n={n}")
    
    total_operators = 4**n
    exhaustive = total_operators <= num_samples

    print(f"\n================ n = {n} (Total space: 4^{n} = {total_operators:,}) ================")
    
    if exhaustive:
        pauli_stream = list(generate_all_paulis(n))
        print(f"Exhaustive test over all {len(pauli_stream)} operators:")
    else:
        pauli_stream = [generate_random_pauli(n) for _ in range(num_samples)]
        print(f"Sampling {num_samples} random operators:")

    start_time = time.perf_counter()
    results = []
    
    for p in pauli_stream:
        x, z = pauli_to_xz(p)
        j = solve_j(x, z, poly=IRREDUCIBLE_POLYS[n])
        results.append((p, j))

    elapsed = time.perf_counter() - start_time
    avg_per_op_ms = (elapsed / len(pauli_stream)) * 1000

    if print_results:
        for p, j in results:
            j_disp = "∞" if (j == float('inf') or np.isinf(j)) else j
            print(f"P = {p}  -->  j = {j_disp}")

    print(f"\nTiming: Total = {elapsed:.4f}s | Avg per operator = {avg_per_op_ms:.3f} ms")

    
def get_basis_states(circuit_or_matrix, n: Optional[int] = None):
    """
    Returns a list of all 2^n statevectors |psi_i^(j)> = U_j |i>,
    which are the columns of the unitary matrix U_j.
    """
    if callable(circuit_or_matrix):
        if n is None:
            raise ValueError("Parameter n must be specified when passing a callable quantum function.")
        U_mat = qml.matrix(circuit_or_matrix, wire_order=list(range(n)))()
    else:
        U_mat = np.asarray(circuit_or_matrix)

    dim = U_mat.shape[1]
    return [U_mat[:, i] for i in range(dim)]


def pauli_string_to_matrix(p_str: str) -> np.ndarray:
    """
    Converts a PennyLane Pauli string like 'IXY' to its 2^n x 2^n matrix representation.
    Leftmost character corresponds to wire 0 (standard PennyLane tensor ordering).
    """
    matrices = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }
    mat = np.array([[1.0]], dtype=complex)
    for char in p_str:
        mat = np.kron(mat, matrices[char])
    return mat


def run_mub_orthogonality_test(n: int, verbose: bool = True):
    poly = IRREDUCIBLE_POLYS[n]
    all_j = list(range(2**n))

    # Precompute all 2^n basis states for each MUB basis j' and standard basis j=inf
    basis_states = {}
    basis_states[float('inf')] = [np.eye(2**n)[:, i] for i in range(2**n)]
    for j_prime in all_j:
        U_j = mub_unitary(n, j_prime)
        basis_states[j_prime] = [U_j[:, i] for i in range(2**n)]

    all_bases = [float('inf')] + all_j
    all_paulis = [
        "".join(p) for p in itertools.product(["I", "X", "Y", "Z"], repeat=n)
        if set(p) != {'I'}
    ]

    print(f"Testing n={n}: {len(all_paulis)} non-identity Paulis across {len(all_bases)} bases...\n")

    for p_str in all_paulis:
        x, z = pauli_to_xz(p_str)
        expected_j = solve_j(x, z, poly=poly)
        P_mat = pauli_string_to_matrix(p_str)

        exp_j_disp = "∞" if (expected_j == float('inf') or np.isinf(expected_j)) else expected_j
        print(f"Pauli P = {p_str:<4} -> Assigned Basis j = {exp_j_disp}")

        for j_prime in all_bases:
            states = basis_states[j_prime]
            vals = [np.vdot(psi, P_mat @ psi).real for psi in states]
            vals_str = ", ".join(f"{v:+.2f}" for v in vals)

            is_match = (j_prime == expected_j)
            tag = "MATCH (±1)" if is_match else "OTHER (0) "

            if verbose:
                basis_label = "∞" if (j_prime == float('inf') or np.isinf(j_prime)) else str(j_prime)
                print(f"  [{tag}] Basis j'={basis_label:<4} | <P> over states: [{vals_str}]")


            # Correctness checks
            for i, val in enumerate(vals):
                if is_match:
                    if not (np.isclose(val, 1.0, atol=1e-5) or np.isclose(val, -1.0, atol=1e-5)):
                        raise AssertionError(
                            f"Mismatch at j'=j={expected_j}: P={p_str}, state={i}, <P>={val:.4f} (expected ±1)"
                        )
                else:
                    if not np.isclose(val, 0.0, atol=1e-5):
                        raise AssertionError(
                            f"Leakage at j'={j_prime} != j={expected_j}: P={p_str}, state={i}, <P>={val:.4f} (expected 0)"
                        )
        print("-" * 60)

    print(f"\n✓ All checks passed for n={n}!")


if __name__ == '__main__':
    run_pauli_solver(n=4, num_samples=10, print_results=True)
    run_mub_orthogonality_test(3, verbose=False)


