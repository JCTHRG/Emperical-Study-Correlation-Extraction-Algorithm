"""
This script implements the Correlation Extraction Algorithm (CEA) using Qiskit.
The algorithm is used to analyze the cryptographic properties of vectorial boolean functions by preparing a quantum
circuit, applying Hadamard gates, and computing statevectors. The script includes functions to apply the Hadamard gate
in parallel, prepare the statevector of the CEA circuit, and execute the CEA algorithm for a given function and number
of rounds.
"""

from math import sqrt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from utils import get_parity
import numba as nb
import numpy as np
import random


@nb.jit(nopython=True, parallel=True)
def cea_apply_hadamard(m, n, state, final_state):
    """
    Custom helper routine for applying an (m+n)-tensored Hadamard gate on a given state with low-level code and in
    parallel.

     Parameters:
        m (int): Number of qubits in the input part of the function.
        n (int): Number of qubits in the output part of the function.
        state (np.ndarray): Initial state vector before applying the Hadamard gate.
        final_state (np.ndarray): State vector after applying the Hadamard gate.

    Returns:
        np.ndarray: Final state vector after applying the Hadamard gate.
    """
    for y in nb.prange(1 << (m + n)):
        for x in range(1 << (m + n)):
            final_state[y] += (-1) ** (get_parity(x & y)) * state[x]
    return final_state


def prepare_cea_circuit_statevector(f):
    """
    Get the circuit statevector of the CEA algorithm for a function `f`.

    Parameters:
        f (BooleanVectorFunction): The vectorial boolean function for which the CEA algorithm statevector is prepared.

    Returns:
        np.ndarray: The statevector of the CEA algorithm for the function `f`.
    """
    (m, n) = (f.m, f.n)
    circuit = QuantumCircuit(m + n, m + n)
    for i in range(m + n):
        circuit.initialize([1, 0], i)
    circuit.h(range(n, m + n))
    state = Statevector(circuit)
    # Multiply by normalization factor for less rounding errors.
    state *= sqrt(1 << (m + n))
    unitary = f.to_sparse_unitary()
    state = unitary.dot(state)
    final_state = np.zeros((1 << (m + n)), dtype=complex)
    cea_apply_hadamard(m, n, state, final_state)
    final_state /= (1 << (m + n))

    return final_state


def cea(f, r, return_statevector=False, seed=None, statevector=None):
    """
    Executes the CEA (Correlation Extraction Algorithm) for a given function `f` with `r` rounds, with Qiskit
    using the seed `seed`.

     Parameters:
        f (BooleanVectorFunction): The vectorial boolean function for which the CEA algorithm is executed.
        r (int): Number of rounds to execute the CEA algorithm.
        return_statevector (bool, optional): If True, return the statevector instead of the mask pairs.
            Default is False.
        seed (int, optional): Seed for the random number generator. Default is None.
        statevector (np.ndarray, optional): Precomputed statevector to use. Default is None.

    Returns:
        list or np.ndarray: If return_statevector is True, returns the statevector. Otherwise, returns a list of mask
        pairs.
    """
    (m, n) = (f.m, f.n)

    if statevector is None:
        statevector = prepare_cea_circuit_statevector(f)

    if return_statevector:
        return statevector

    random.seed(seed)
    result = random.choices(range(1 << (m + n)), weights=[abs(a) ** 2 for a in statevector], k=r)
    mask_pairs = [(x >> n, x - (x & (((1 << (m + n)) - 1) ^ ((1 << n) - 1)))) for x in result]
    return mask_pairs
