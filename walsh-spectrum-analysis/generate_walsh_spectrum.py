"""
This script generates the Walsh spectrum for cryptographic functions and writes the results to CSV files.

Key functionalities include:
1. Calculating the Walsh transform for input-output mask pairs of cryptographic functions.
2. Counting the occurrences of each Walsh spectrum value.
3. Writing the results to CSV files for further analysis.

The script is designed to analyze the Walsh spectrum for different cryptographic setups, such as the XOR cipher with a
fixed key and the AES S-box, and outputs the results in CSV format.

Usage:
- The script can be executed directly to generate Walsh spectrum data for predefined cryptographic functions.
- The `generate_walsh_spectrum` function calculates the Walsh transform and writes the results to a CSV file.
"""

import csv
from functions.rijndael_s_box import rijndael_s_box
from functions.xor_cipher_with_fixed_key import XORCipherWithFixedKey
from utils import *
from collections import defaultdict


def generate_walsh_spectrum(file_name, f):
    """
    Generates the Walsh spectrum for a given cryptographic function and writes the results to a CSV file.

    Parameters:
    file_name (str): The name of the CSV file to store the results.
    f (BooleanVectorFunction): The cryptographic function to be analyzed.

    The function calculates the Walsh transform for all input-output mask pairs, counts the occurrences
    of each Walsh spectrum value, and writes the results to the specified CSV file.
    """
    walsh_spec = defaultdict(int)
    with open(file_name, "w") as out:
        writer = csv.writer(out)
        for a in range(2 ** f.m):
            for b in range(2 ** f.n):
                wt = walsh_transform(f, a, b)
                walsh_spec[wt] += 1
                print(a, ", ", b, " - ", wt, "/", 256 ** 2, sep="")
                writer.writerow([a, b, wt])


if __name__ == '__main__':
    # Generate Walsh spectrum for XOR cipher with fixed key and save to CSV
    generate_walsh_spectrum('xor_cipher_with_fixed_key.csv', XORCipherWithFixedKey(8, ord('Y')))
    # Generate Walsh spectrum for AES S-box and save to CSV
    generate_walsh_spectrum('rijndael_s_box.csv', rijndael_s_box)
