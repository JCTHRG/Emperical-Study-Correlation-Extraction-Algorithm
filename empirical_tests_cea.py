"""
This script conducts empirical tests for the Cryptographic Exhaustive Analysis (CEA) on different cryptographic
functions.
It includes experiments to measure time and space usage, as well as empirical studies to collect data on mask pairs
and their characteristics such as Walsh transform and bias. The results are serialized and stored in binary files for
further analysis.

Key functionalities include:
1. Measuring time and memory usage of CEA for a specified cipher.
2. Conducting empirical studies to gather data on mask pairs, their Walsh transform, and bias.
3. Serializing and storing the results in binary files.
4. Reading and parsing the results from the binary files.

The script is executed with two main empirical studies:
- Analysis of the Rijndael S-box.
- Analysis of the XOR cipher with a fixed key.

Usage:
- The script can be executed directly to run the predefined empirical studies.
- The `exp_time_space_usage` function can be used to measure the performance of CEA for a specified cipher.
- The `exp_empirical_study` function collects and stores empirical data for further analysis.
- The `get_exp_results` function reads and parses the results from the binary files produced by the empirical studies.
"""

from cea import prepare_cea_circuit_statevector, cea
from functions.rijndael_s_box import rijndael_s_box
from functions.xor_cipher_with_fixed_key import XORCipherWithFixedKey
from utils import *
from math import e, floor, pi
import struct
import time
import tracemalloc

BASE_SEED_PI = floor(pi * 1e3) + 0xCA684ACA
BASE_SEED_E = floor(e * 1e3) + 0xABA640E0


def exp_time_space_usage(f=XORCipherWithFixedKey(8, ord('Y'))):
    """
    Small experiment to test the time and space usage of CEA for a cipher with m+n=16.

    Parameter:
        f (BooleanVectorFunction): The cryptographic function to be executed.

    Returns:
        None
    """
    tracemalloc.start()
    t1 = round(time.time() * 1000)
    statevector = prepare_cea_circuit_statevector(f)
    cea(f, 1000, statevector=statevector)
    t2 = round(time.time() * 1000)
    print(f"Total Time:        {t2 - t1} ms")
    print(f"Highest RAM Usage: {round(tracemalloc.get_traced_memory()[1] / 1024 ** 3, 4)} GiB")
    tracemalloc.stop()


def exp_empirical_study(file_name, f=XORCipherWithFixedKey(8, ord('Y')), seed=BASE_SEED_PI, tau=1 / 16,
                        batch_size=1 << 10, rounds=1000):
    """
    Execute the empirical study.
    This function collects data on mask pairs, their Walsh transform, and bias, and stores the results in a binary file.

    Parameters:
       file_name (str): The name of the file to store the results.
       f (BooleanVectorFunction): The cryptographic function to be analyzed.
       seed (int): The initial seed for random number generation.
       tau (float): The threshold for the bias.
       batch_size (int): The number of samples in each batch.
       rounds (int): The number of rounds to run the study.

    Returns:
        None
    """
    statevector = prepare_cea_circuit_statevector(f)
    # Create and clear file
    open(file_name, "wb").close()
    with open(file_name, "ab") as file:
        for i in range(rounds):
            print(i, end=" ", flush=True)
            found = False
            while not found:
                seed += 1
                mask_pairs = cea(f, batch_size, seed=seed, statevector=statevector)
                for (a, b) in mask_pairs:
                    # Compute quality of characteristic
                    mec = mask_equality_count(f, a, b)
                    wa = walsh_transform(f, a, b, mec=mec)
                    # Due to rounding, some mask pairs with Walsh Transform of zero may be sampled, just skip those.
                    if wa == 0:
                        continue
                    bi = bias(f, a, b, mec=mec)
                    # Serialize
                    # File format: Each mask uses 8 bit, so a bytes each. w and b can be stored as 32-bit floats, so one
                    # entry requires 10 byte. We store the ten bits sequentially. We additionally store the round
                    # index i at the start (4 byte).
                    file.write(
                        struct.pack("i", i) + bytes([a, b]) + struct.pack("f", wa) + struct.pack("f", bi)
                    )
                    # Check if found
                    if abs(bi) >= tau and (a, b) != (0, 0):
                        found = True
                        break
    print()


def get_exp_results(file_name):
    """
    Get results from a file produced by `exp_empirical_study`.
    The result is a list of quadruples with the input and output masks, as well as their walsh transform and bias,
    in this order.

    Parameters:
        file_name (str): The name of the file containing the results.

    Returns:
        list: A list of tuples containing the round index, input mask, output mask, walsh transform, and bias.
    """
    results = []
    with open(file_name, "rb") as file:
        by = file.read(14)
        while len(by) == 14:
            i = int(struct.unpack("i", by[0:4])[0])
            a = by[4]
            b = by[5]
            wa = int(struct.unpack("f", by[6:10])[0])
            bi = struct.unpack("f", by[10:14])[0]
            results.append((i, a, b, wa, bi))
            by = file.read(14)
    return results


if __name__ == '__main__':
    #    # Main study experiments
    exp_empirical_study("rijndael_s_box.bin", f=rijndael_s_box, seed=BASE_SEED_PI, rounds=3000)
    exp_empirical_study(
        "xor_cipher_with_fixed_key.bin", f=XORCipherWithFixedKey(8, ord('Y')), seed=BASE_SEED_E, rounds=3000
    )
