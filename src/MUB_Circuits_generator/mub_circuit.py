from typing import List, Tuple
import numpy as np
from qiskit import QuantumCircuit

from .consts import IRREDUCIBLE_POLYS

def to_base_p(x: int, p: int, n: int) -> np.ndarray:
    digits = []
    for _ in range(n):
        digits.append(x % p)
        x //= p
    return np.array(digits, dtype=int)

def from_base_p(digits: np.ndarray, p: int) -> int:
    x = 0
    for i, d in enumerate(digits):
        x += int(d) * (p ** i)
    return x

def binary_to_vector(x: int) -> np.ndarray:
    return np.fromiter(f'{x:b}', dtype=int)[::-1]

def calculate_galois_matrices(n: int) -> Tuple[List[np.ndarray], List[int]]:
    irreducible_poly = IRREDUCIBLE_POLYS[n]
    p_powers = []
    for index in range(n):
        x = np.zeros(n, dtype=int)
        x[index] = 1
        p_powers.append(x)
    for extended_index in range(n - 1):
        prev_x = p_powers[-1]
        x = np.concatenate(([0], prev_x[:-1]))
        x += prev_x[-1] * binary_to_vector(irreducible_poly)[:-1]
        p_powers.append(x)

    galois_matrices = []
    for index in range(n):
        galois_matrix = np.zeros((n, n), dtype=int)
        for s in range(n):
            for t in range(n):
                galois_matrix[s][t] = p_powers[s + t][index]
        galois_matrices.append(galois_matrix)
    return galois_matrices, [from_base_p(x, 2) for x in p_powers]

def gf_mul(a: int, b: int, galois_matrices: List[np.ndarray], n: int, lsb_bits: int = None) -> int:
    if lsb_bits is None:
        lsb_bits = n
    else:
        lsb_bits = min(n, lsb_bits)
        
    out = 0
    a_v = to_base_p(a, 2, n)
    b_v = to_base_p(b, 2, n)
    for index in range(lsb_bits):
        galois_matrix = galois_matrices[index]
        index_val = (a_v.T @ galois_matrix @ b_v) % 2
        out += index_val.item() * (1 << index)
    return out

# def calculate_a(j: int, galois_matrices: List[np.ndarray], p_powers: List[int], n: int) -> np.ndarray:
#     a_arr = np.zeros(n, dtype=int)
#     for index in range(n):
#         inner_term = p_powers[2*index]
#         exponent = gf_mul(j, inner_term, galois_matrices, n, lsb_bits=2)
#         a_arr[index] = exponent if exponent % 2 == 0 else 4 - exponent  # take the conjugate into account
#     return a_arr

# def calculate_b(j: int, galois_matrices: List[np.ndarray], p_powers: List[int], n: int) -> np.ndarray:
#     b_arr = np.zeros(2 * n - 1, dtype=int)
#     for index in range(2 * n - 1):
#         inner_term = p_powers[index]
#         b_arr[index] = gf_mul(j, inner_term, galois_matrices, n, lsb_bits=1)
#     return b_arr

def _gf2_multiply(a: int, b: int, poly: int, n: int) -> int:
    """Multiplication in GF(2^n) modulo irreducible polynomial `poly`."""
    res = 0
    poly_rem = poly ^ (1 << n)  # reduction polynomial without x^n
    for _ in range(n):
        if b & 1:
            res ^= a
        b >>= 1
        carry = a & (1 << (n - 1))
        a = (a << 1) & ((1 << n) - 1)
        if carry:
            a ^= poly_rem
    return res



def compute_field_trace(beta: int, poly: int, n: int) -> int:
    """
    Computes Tr(beta) = sum_{m=0}^{n-1} beta^(2^m) in GF(2^n).
    Returns 0 or 1.
    """
    trace_val = 0
    curr = beta
    for _ in range(n):
        trace_val ^= curr
        curr = _gf2_multiply(curr, curr, poly, n)  # Frobenius squaring
    return trace_val & 1

# IMPORTANT: I changed how a,b are calculated, I think this is better.
# also added trace_table

def get_trace_table(n: int) -> np.ndarray:
    """Precompute Tr(alpha^m) mod 2 using the canonical field trace."""
    poly = IRREDUCIBLE_POLYS[n]
    max_power = 3 * n
    traces = np.zeros(max_power, dtype=np.uint8)
    
    beta = 1  # alpha^0 = 1
    alpha = 2 # alpha^1 is 0b0010 in standard representation
    for m in range(max_power):
        traces[m] = compute_field_trace(beta, poly, n)
        beta = _gf2_multiply(beta, alpha, poly, n)
        
    return traces

def calculate_a(j, n: int, trace_table: np.ndarray) -> np.ndarray:
    # 1. Handle computational basis (j = infinity)
    if j == float('inf') or np.isinf(j):
        return np.zeros(n, dtype=int)

    # 2. Ensure j and k are standard Python ints
    j_int = int(j)
    j_bits = [(j_int >> k) & 1 for k in range(n)]
    
    a_arr = np.zeros(n, dtype=int)
    for s in range(n):
        val = sum(j_bits[k] * int(trace_table[int(k + 2 * s)]) for k in range(n)) % 2
        a_arr[s] = int(val)
    return a_arr

def calculate_b(j, n: int, trace_table: np.ndarray) -> np.ndarray:
    # 1. Handle computational basis (j = infinity)
    if j == float('inf') or np.isinf(j):
        return np.zeros(2 * n - 1, dtype=int)

    # 2. Ensure j and k are standard Python ints
    j_int = int(j)
    j_bits = [(j_int >> k) & 1 for k in range(n)]
    
    b_arr = np.zeros(2 * n - 1, dtype=int)
    for m in range(2 * n - 1):
        val = sum(j_bits[k] * int(trace_table[int(k + m)]) for k in range(n)) % 2
        b_arr[m] = int(val)
    return b_arr

def mub_circuit(n: int, j: int) -> QuantumCircuit:
    """
    Build the MUB circuit U(j) for n qubits.
    The circuit does H^⊗n, then S^a on each qubit, then CZ gates.
    """
    qc = QuantumCircuit(n, name=f"MUB generator for j={j}")
    # H-part: Hadamard on all qubits
    for q in range(n):
        qc.h(q)
    # Compute S and CZ parameters for this j
    galois_matrices, p_powers = calculate_galois_matrices(n)
    trace_table = get_trace_table(n)
    a = calculate_a(j, n, trace_table)
    b = calculate_b(j, n, trace_table)
    # print(f"a: {a}\n b: {b}")
    # S-part: apply S^a_t on qubit t
    for t in range(n):
        # a[t] == 0: do nothing
        if a[t] == 1:
            qc.s(t)
        elif a[t] == 2:
            qc.z(t)
        elif a[t] == 3:
            qc.sdg(t)
    # CZ-part: apply CZ(s,t) where b[(s,t)] == 1
    for s in range(n):
        for t in range(s + 1, n):
            if b[s + t] == 1:
                qc.cz(s, t)
    return qc