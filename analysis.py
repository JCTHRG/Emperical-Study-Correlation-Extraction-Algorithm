"""
This script analyzes experimental results related to the probability of masks with specific Walsh transforms
in cryptographic applications. It calculates theoretical probabilities and Clopper-Pearson confidence intervals
for the success probability of the CEA experiments and saves the results in JSON files. The script supports two main
analyses: one for the XOR cipher with a fixed key and one for the AES S-Box.

Functions:
- write_object_to_json_file: Saves a Python object to a JSON file.
- read_from_json_file: Reads a Python object from a JSON file.
- sbox_p_0: Calculates the probability of CEA for a mask with a Walsh transform of +-32 for the AES S-Box.
- xor_p_0: Calculates the probability of CEA for a mask with a Walsh transform of 2 ** 8 for the XOR cipher.
- get_k: Computes the cumulative count of experiments that finished in r or fewer measurements.
- group_experiments: Groups experiments based on the number of measurements.
- create_count_dict: Creates a dictionary counting the number of experiments that finished after a certain number
    of measurements.
"""

from empirical_tests_cea import get_exp_results
import scipy.stats as stats
from collections import defaultdict
import json


KEY_THEORETICAL_PROB = "theoretical probability"
KEY_CONFIDENCE_LOWER_BOUND = "confidence lower bound"
KEY_CONFIDENCE_UPPER_BOUND = "confidence upper bound"
KEY_MEASURED_PROB = "measured probability"


def write_object_to_json_file(file_name, obj):
    """
    Write a Python object to a JSON file.

    Parameters:
        file_name (str): The name of the file to write the JSON data to.
        obj (any): The Python object to serialize to JSON and write to the file.

    Returns:
        None
    """
    # Create and clear file
    open(file_name, "wb").close()
    with open(file_name, 'w') as json_file:
        json.dump(obj, json_file, indent=4)


def read_from_json_file(file_name):
    """
    Read a Python object from a JSON file.

    Parameters:
        file_name (str): The name of the file to read the JSON data from.

    Returns:
        any: The Python object deserialized from the JSON file.
    """
    with open(file_name, 'r') as json_file:
        obj = json.load(json_file)
    return obj


def sbox_p_0(r):
    """
    This calculates the probability for CEA for the AES sbox to measure a mask with a Walsh transform of +-32.

    Parameter:
        r (int): Number of CEA measurements

    Return:
        (float) Probability to for r measurement
    """
    # According to our Walsh spectrum analysis of the rjindael_s_box there are 1275 mask with an absolute Walsh
    # transform of 32
    wf = 32
    cf = wf / (2 ** 8)
    cfs = cf ** 2
    num_masks_with_wf = 1275
    return 1 - (1 - ((num_masks_with_wf * cfs) / 2 ** 8)) ** r


def xor_p_0(r):
    """
    This calculates the probability for CEA for the XOr cipher with fixed key to measure a mask with a Walsh
    transform of 2 ** 8.

    Parameter:
        r (int): Number of CEA measurements

    Return:
        float: Probability to for r measurement
    """
    wf = 2 ** 8
    cf = wf / 2 ** 8
    cfs = cf ** 2
    num_masks_with_wf = 2 ** 8 - 1
    return 1 - (1 - ((num_masks_with_wf * cfs) / 2 ** 8)) ** r


def get_k(counts, r):
    """
     Calculate the cumulative count of experiments that finished in r or fewer measurements.

     Parameters:
         counts (dict): Dictionary where keys are the number of measurements and values are the counts of experiments.
         r (int): The number of measurements up to which to count.

     Returns:
         int: The cumulative count of experiments up to r measurements.
     """
    count = 0
    for i in range(r + 1):
        count += counts.get(i, 0)
    return count


def group_experiments(cea_measurements):
    """
   Group the list of experiments by their m_number number.

   Parameters:
        cea_measurements (list): List of tuples, each containing (m_number, a, b, wf, bi), where
            - m_number is the number identifying measurement of the experiment, e.g., m_number=1 for the first
            measurement
            - (a, b) is the measured mask
            - wf is the Walsh transform of (a, b)
            - bi is the bias of (a, b)

    Returns:
        defaultdict: A dictionary where keys are m_number numbers and values are lists of experiments.
   """
    experiments = defaultdict(list)
    # Grouping of experiments
    for m_number, a, b, wf, bi in cea_measurements:
        experiments[m_number].append((a, b, wf, bi))
    return experiments


def create_count_dict(experiments, t_num_exp):
    """
   Create a dictionary counting the number of experiments that finished after a certain number of measurements.

    Parameters:
        experiments (dict): Dictionary where keys are m_number numbers and values are lists of experiments.
        t_num_exp (int): Total number of experiments.

    Returns:
        tuple: A dictionary of counts and a list of r values.
    """
    counts = defaultdict(int)
    r_values = []
    # Create sample count dict counting the number of experiment finished after a certain number of measurements
    for m_number in range(t_num_exp):
        r_value = len(experiments.get(m_number, []))
        if r_value not in r_values:
            r_values.append(r_value)
        counts[r_value] += 1
    return counts, r_values


def clopper_pearson_interval(k, n, alpha):
    """
    Calculate the Clopper-Pearson confidence interval for a binomial proportion.

    Parameters:
        k (int): Number of successes.
        n (int): Total number of trials.
        alpha (float): Significance level.

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    p_l = 0.0
    # lower bound p_L
    if k != 0:
        p_l = stats.beta.ppf(alpha / 2, k, n - k + 1)
    # Upper bound p_U
    if k == n:
        p_u = 1.0
    else:
        p_u = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return p_l, p_u


def confidence_intervals(experiments_file_name, alpha, p0_func, json_file_name):
    """
    Calculate and print confidence intervals for the probability of success in experiments.

    Parameters:
       experiments_file_name (str): Filename of the experiments data.
       alpha (float): Significance level for the confidence intervals.
       p0_func (BooleanVectorFunction): Function to calculate the theoretical probability pi_0.
       json_file_name (str): Filename to save the results in JSON format.

    Returns:
       None
    """
    results = get_exp_results(experiments_file_name)

    count = 0
    for i, a, b, wf, bi in results:
        if a == 0 and b == 0:
            count += 1
    print("Measured {} trivial masks".format(count))

    experiments = group_experiments(results)
    print("Number of experiments: {}".format(len(experiments)))

    t_num_exp = len(experiments)
    counts, r_values = create_count_dict(experiments, t_num_exp)
    r_values.sort()
    overall_max_dist = 0.0
    overall_min_dist = None

    pi0s = []
    p_ls = []
    p_us = []
    ps = []
    for r in r_values:
        k = get_k(counts, r)
        (p_L, p_U) = clopper_pearson_interval(k, t_num_exp, alpha)
        p_ls.append((r, p_L))
        p_us.append((r, p_U))
        pi_0 = p0_func(r)
        pi0s.append((r, pi_0))
        p = k / t_num_exp
        ps.append((r, p))
        p0_in_interval = p_L <= pi_0 <= p_U
        p_in_interval = p_L <= p <= p_U
        max_dist = max(abs(p_L - pi_0), abs(p_U - pi_0))
        min_dist = min(abs(p_L - pi_0), abs(p_U - pi_0))
        overall_max_dist = max(max_dist, overall_max_dist)
        if overall_min_dist is None:
            overall_min_dist = min_dist
        else:
            overall_min_dist = min(min_dist, overall_min_dist)
        print(
            "r: {}, confidence interval: {}, pi_0: {}, pi_0 in (p_l,p_u): {}, p in (p_l,p_u: {} max_dist: {},"
            "min_dist: {}, k: {}".format(
                r,
                (p_L, p_U),
                pi_0,
                p0_in_interval,
                p_in_interval,
                max_dist,
                min_dist,
                k
            )
        )
    coordinates = defaultdict(list)
    coordinates[KEY_THEORETICAL_PROB] = pi0s
    coordinates[KEY_CONFIDENCE_LOWER_BOUND] = p_ls
    coordinates[KEY_CONFIDENCE_UPPER_BOUND] = p_us
    coordinates[KEY_MEASURED_PROB] = ps

    write_object_to_json_file(json_file_name, coordinates)

    print("Overall min dist: {}".format(overall_min_dist))
    print("Overall max dist: {}".format(overall_max_dist))


if __name__ == '__main__':
    file_name_xor = 'analysis-results/xor_cipher_with_fixed_key_conf_coords.json'
    confidence_intervals(
        'experiment-results/xor_cipher_with_fixed_key.bin',
        0.01, xor_p_0,
        file_name_xor
    )

    file_name_s_box = 'analysis-results/rijndael_s_box_conf_coords.json'
    confidence_intervals(
        'experiment-results/rijndael_s_box.bin',
        0.01,
        sbox_p_0,
        file_name_s_box
    )
