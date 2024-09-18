"""
This script performs an analysis of the Walsh spectrum for cryptographic data contained in CSV files and prints the
results.

Key functionalities include:
1. Reading the Walsh spectrum data from CSV files.
2. Counting the occurrences of each Walsh spectrum value.
3. Sorting the Walsh spectrum values.
4. Printing the Walsh spectrum values.

The script is designed to analyze the Walsh spectrum for different cryptographic setups, such as the XOR cipher with a
fixed key and the AES S-box, and outputs the results in a graphical format for further analysis.

Usage:
- The script reads data from CSV files specified in the `walsh_spectrum_analysis` function calls in the `__main__`
    section.
- It processes the CSV files to count and sort the Walsh spectrum values.

Example:
- The script analyzes the Walsh spectrum for "xor_cipher_with_fixed_key.csv" and "rijndael_s_box.csv".
"""
import csv


def walsh_spectrum_analysis(file_name):
    """
    Analyzes the Walsh spectrum from a CSV file and prints the spectrum.

    Parameters:
    file_name (str): The name of the CSV file containing the Walsh spectrum data.

    The function reads the CSV file, counts the occurrences of each Walsh spectrum value,
    sorts the spectrum, and then prints the sorted spectrum values.
    """
    walsh_spectrum = {}

    # Read the CSV file and count the occurrences of each Walsh spectrum value
    with open(file_name) as outfile:
        reader = csv.reader(outfile)
        for row in reader:
            if row[2] in walsh_spectrum.keys():
                walsh_spectrum[row[2]] += 1
            else:
                walsh_spectrum[row[2]] = 1

    # Sort the Walsh spectrum by keys (converted to integers)
    t = {}
    for key in sorted(walsh_spectrum.keys(), key=lambda x: int(x)):
        t[key] = walsh_spectrum[key]
    walsh_spectrum = t

    print("Walsh spectrum for {}: {}".format(file_name, walsh_spectrum.items()))


if __name__ == '__main__':
    # Analyze the Walsh spectrum for the provided CSV files
    walsh_spectrum_analysis("xor_cipher_with_fixed_key.csv")
    walsh_spectrum_analysis("rijndael_s_box.csv")
