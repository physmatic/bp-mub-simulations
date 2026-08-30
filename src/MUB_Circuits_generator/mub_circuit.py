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

def calculate_a(j: int, galois_matrices: List[np.ndarray], p_powers: List[int], n: int) -> np.ndarray:
    a_arr = np.zeros(n, dtype=int)
    for index in range(n):
        inner_term = p_powers[2*index]
        exponent = gf_mul(j, inner_term, galois_matrices, n, lsb_bits=2)
        a_arr[index] = exponent if exponent % 2 == 0 else 4 - exponent  # take the conjugate into account
    return a_arr

def calculate_b(j: int, galois_matrices: List[np.ndarray], p_powers: List[int], n: int) -> np.ndarray:
    b_arr = np.zeros(2 * n - 1, dtype=int)
    for index in range(2 * n - 1):
        inner_term = p_powers[index]
        b_arr[index] = gf_mul(j, inner_term, galois_matrices, n, lsb_bits=1)
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
    a = calculate_a(j, galois_matrices, p_powers, n)
    b = calculate_b(j, galois_matrices, p_powers, n)
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