"""
This script defines a class `XORCipherWithFixedKey` which represents a simple XOR cipher with a fixed key.
The class inherits from `BooleanVectorFunction` and initializes with a given key `k` and input size `m`.
The XOR cipher is a basic encryption algorithm where each bit of the plaintext is XORed with the corresponding bit of
the key to produce the ciphertext.

Classes:
    - XORCipherWithFixedKey: Represents the XOR cipher with a fixed key.
"""

from utils import BooleanVectorFunction


class XORCipherWithFixedKey(BooleanVectorFunction):
    def __init__(self, m, k):
        """
        Initializes the XOR cipher with a fixed key.

        Parameters:
           m (int): The size of the input and output vectors.
           k (int): The fixed key to be used for the XOR operation.
        """
        def xor(x):
            return x ^ k
        BooleanVectorFunction.__init__(self, xor, m, m)
        del xor
