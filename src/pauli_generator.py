from typing import List, Tuple
from .mub_circuit import calculate_galois_matrices, gf_mul

def generate_pauli_sets(n: int) -> List[List[str]]:
    """
    Generates the (2^n + 1) maximal commuting Pauli sets for n qubits
    using the Galois field GF(2^n) construction.
    
    Returns:
        A list of lists of Qiskit Pauli string representations.
        Index 0 corresponds to the Standard Basis (Z-basis).
        Indices 1 to 2^n correspond to the MUB circuits with parameter j=0 to 2^n-1.
    """
    galois_matrices, _ = calculate_galois_matrices(n)
    
    def to_pauli_char(a_bit: int, b_bit: int) -> str:
        if a_bit == 0 and b_bit == 0: return 'I'
        if a_bit == 1 and b_bit == 0: return 'X'
        if a_bit == 0 and b_bit == 1: return 'Z'
        if a_bit == 1 and b_bit == 1: return 'Y'
        raise ValueError("Bits must be 0 or 1")
    
    def construct_pauli_string(a: int, b: int) -> str:
        """
        Constructs the Qiskit Pauli string for the operator (a, b).
        Qiskit uses right-to-left endianness: qubit 0 is the rightmost character.
        """
        chars = []
        for qubit in range(n):
            a_bit = (a >> qubit) & 1
            b_bit = (b >> qubit) & 1
            chars.append(to_pauli_char(a_bit, b_bit))
        # Reverse because Qiskit string has qubit 0 at the rightmost index
        return "".join(reversed(chars))

    # There are 2^n + 1 bases
    dim = 2 ** n
    all_sets = []

    # 1. Basis 0 (Standard Basis): Z-basis stabilizers (0, b) for b != 0
    c0_set = []
    for b in range(1, dim):
        c0_set.append(construct_pauli_string(0, b))
    all_sets.append(c0_set)

    # 2. Bases 1 to 2^n (MUB j=0 to 2^n-1): (a, j * a) for a != 0
    for j in range(dim):
        cj_set = []
        for a in range(1, dim):
            # Galois field multiplication to get Z-components
            b = gf_mul(j, a, galois_matrices, n, lsb_bits=n)
            cj_set.append(construct_pauli_string(a, b))
        all_sets.append(cj_set)

    return all_sets

if __name__ == "__main__":
    # Quick test for n=2
    sets_n2 = generate_pauli_sets(2)
    for i, s in enumerate(sets_n2):
        print(f"C_{i}: {s}")
