"""
Script for modeling and analyzing Boolean vector functions.

This script provides a class `BooleanVectorFunction` along with several utility functions to handle Boolean vector
functions. It includes methods to convert integers to binary vectors, one-hot encoded vectors, and vice versa.
The script also contains functions to compute parity, mask equality counts, Walsh transforms, correlations, and biases
for vectorial boolean functions. Additionally, it leverages the `scipy.sparse` module to create sparse unitary matrices
and `numba` for optimized parity computation.

Modules:
- `bitarray`: Used for efficient bitwise operations.
- `scipy.sparse`: Provides functionality to create sparse matrices.
- `numba`: Used to optimize specific functions with JIT compilation.

Key Classes and Functions:
- `BooleanVectorFunction`: Class that models Boolean vector functions and precomputes bitarray representations.
- `integer_to_np_one_hot_encoded`: Converts an integer to a one-hot encoded vector.
- `integer_to_binary_vector`: Converts an integer to a binary vector.
- `binary_vector_to_integer`: Converts a binary vector to an integer.
- `get_parity`: Computes the parity of an integer.
- `mask_equality_count`: Computes the number of inputs that satisfy a specific mask equality condition.
- `mask_equality_probability`: Computes the probability of the mask equality condition.
- `walsh_transform`: Computes the Walsh transform of a vectorial Boolean function.
- `correlation`: Computes the correlation of a vectorial Boolean function.
- `bias`: Computes the bias of a vectorial Boolean function.

Note:
- The `tweedledum` library is not used due to installation issues, and hence a custom `to_sparse_unitary` method is
provided.
"""

from bitarray import bitarray
from scipy.sparse import coo_array
import numba as nb

MAX_BIT_COUNT = 16


class BooleanVectorFunction:
    """
    Class for modeling Boolean vector functions.

    Attributes:
        f (bitarray): Precomputed bitarray representing the function.
        m (int): Number of input bits.
        n (int): Number of output bits.
    """

    def __init__(self, f, m, n):
        """
        Initialize the Boolean vector function.

        Parameters:
            f (function): A function that maps an integer to another integer.
            m (int): Number of input bits.
            n (int): Number of output bits.

        The function precomputes the bitarray representation of the function to avoid duplicate computation.
        """
        assert (m <= MAX_BIT_COUNT and n <= MAX_BIT_COUNT)
        # Precompute f to avoid duplicate computation.
        # Scripts are to be executed once, so this will not cause a performance issue.
        self.f = bitarray((1 << m) * n)
        print(f"Allocated {(1 << m) * n} bits")
        for x in range(1 << m):
            self.f[x * n:(x + 1) * n] = bitarray(format(f(x), f"#0{n + 2}b")[2:])
        self.__call__ = lambda x: int(self.f[(x * n):((x + 1) * n)].to01(), 2)
        self.m = m
        self.n = n

    # Custom `to_gate` function, as `tweedledum` does not build with `pip` for some reason, as of 05.07.2024, 00:14 with
    # version v1.1.1., so Qiskits `classical_function` API cannot be used.
    # See also
    # https://docs.quantum.ibm.com/api/qiskit/classicalfunction and https://github.com/boschmitt/tweedledum/issues/186.
    def to_sparse_unitary(self):
        """
        Creates a `scikit.sparse.coo_array` corresponding to an oracle unitary for the function.

        Returns:
            coo_array: A sparse unitary matrix representing the oracle unitary for the function.
        """

        def col_generator(f):
            for x in range(1 << f.m):
                for y in range(1 << f.n):
                    # Determines index for setting one according to
                    # the oracle description |x>|y> |-> |x>|y + f(x)>.
                    yield (x << f.n) | (y ^ f.__call__(x))

        size = 1 << (self.m + self.n)
        unitary = coo_array(([1] * size, (list(col_generator(self)), range(size))), shape=(size, size),
                            dtype=complex).tocsc()
        return unitary


def integer_to_np_one_hot_encoded(x, n):
    """
    Converts an integer to a one-hot encoded vector.

    Parameters:
        x (int): The integer to convert.
        n (int): The length of the one-hot encoded vector.

    Returns:
        list: A one-hot encoded vector.
    """
    assert (0 <= x < n)
    return [0] * x + [1] + [0] * (n - x - 1)


def integer_to_binary_vector(x, n):
    """
    Converts an integer to a binary vector.

    Parameters:
        x (int): The integer to convert.
        n (int): The length of the binary vector.

    Returns:
        list: A binary vector.
    """
    assert (0 <= x < 1 << n)
    return [(x & (1 << i)) >> i for i in reversed(range(n))]


def binary_vector_to_integer(b):
    """
    Converts a binary vector to an integer.

    Parameters:
        b (list): The binary vector to convert.

    Returns:
        int: The resulting integer.
    """
    s = 0
    for b_ in b:
        s += b_
        s = s << 1
    s = s >> 1
    return s


@nb.jit
def get_parity(x):
    """
    Optimized parity computation for a number `x`.

    Parameters:
        x (int): The integer to compute the parity for.

    Returns:
        int: The parity of the integer.
    """
    r = 0
    while x != 0:
        r ^= (x & 1)
        x >>= 1
    return r


def mask_equality_count(f, a, b):
    """
   Computes the number of x that satisfy the condition <x, a> + <f(x), b> = 0 for a given function f and masks a, b.

   Parameters:
        f (BooleanVectorFunction):
            The vectorial Boolean function for which the condition is evaluated.
        a (int):
            Mask applied to the input x.
        b (int):
            Mask applied to the output f(x).

    Returns:
        int: The count of x that satisfy the mask equality condition.
   """
    (m, n) = (f.m, f.n)
    c = 0
    for x in range(1 << m):
        y = f.__call__(x)
        e = get_parity(x & a) + get_parity(y & b)
        c += (e % 2 == 0)
    return c


def mask_equality_probability(f, a, b, mec=None):
    """
    Divides the mask_equality_count(f, a, b) by (1<<f.m).

    Parameters:
        f (BooleanVectorFunction):
            The vectorial Boolean function for which the condition is evaluated.
        a (int):
            Mask applied to the input x.
        b (int):
            Mask applied to the output f(x).
        mec (int), optional:
            Precomputed mask equality count.

    Returns:
        float: The probability of the mask equality condition.
    """
    if mec is None:
        p = mask_equality_count(f, a, b) / (1 << f.m)
    else:
        p = mec / (1 << f.m)
    return p


def walsh_transform(f, a, b, mec=None):
    """
    Computes the Walsh transform.

    Parameters:
        f (BooleanVectorFunction):
            The vectorial Boolean function for which the Walsh transform is evaluated.
        a (int):
            Mask applied to the input x.
        b (int):
            Mask applied to the output f(x).
        mec (int), optional:
            Precomputed mask equality count.

    Returns:
        int: The value of the Walsh transform
    """
    if mec is None:
        e = mask_equality_count(f, a, b)
    else:
        e = mec
    w = e - ((1 << f.m) - e)
    return w


def correlation(f, a, b, mec=None):
    """
    Computes the correlation.

    Parameters:
        f (BooleanVectorFunction):
            The vectorial Boolean function for which the correlation is evaluated.
        a (int):
            Mask applied to the input x.
        b (int):
            Mask applied to the output f(x).
        mec (int), optional:
            Precomputed mask equality count.

    Returns:
        float: The correlation value.
    """
    corr = walsh_transform(f, a, b, mec) / (1 << f.m)
    return corr


def bias(f, a, b, mec=None):
    """
    Computes the bias.

    Parameters:
        f (BooleanVectorFunction):
            The vectorial Boolean function for which the bias is evaluated.
        a (int):
            Mask applied to the input x.
        b (int):
            Mask applied to the output f(x).
        mec (int), optional:
            Precomputed mask equality count.

    Returns:
        float: The bias value.
    """
    return correlation(f, a, b, mec) / 2
