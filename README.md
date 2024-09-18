# Empirical Tests for the Paper "Quantum Search of Linear Approximations: an Extended Analysis of the Correlation Extraction Algorithm"

## Setup

We have tested the software on an Arch Linux system, as of the 9th July 2024, 22:59, with Python 3.12.4.

Clone the repository, navigate into the root directory and setup plus enter a virtual environment by
```
$ python -m venv ../empirical-tests-malviya-paper-venv
$ source empirical-tests-malviya-paper-venv/bin/activate.fish
$ pip install -r requirements.txt
```
when using the `fish` shell. Experiments can be executed using the primitives explained below. To execute a simple benchmark with an 8x8 random vectorial Boolean function, execute:
```
$ python -i empirical_tests_cea.py
>>> exp_time_space_usage(f=RandomCipher(8, 8))
>>> exit()
```

## Design of the Implementation for the Study

The central file for executing the experiments is `empirical_tests_cea.py`. There, a benchmark for time and memory of the CEA simulation and the study code are implemented.
The main file for analyzing the experiments is `analysis.py`, which calculates the confidence intervals.

In `utils.py`, a wrapper for mxn vectorial Boolean functions `BooleanVectorFunction` is defined to give a uniform interface for specifying such functions and their dimension, as well as to convert them to quantum gates. Several such Boolean vector functions have been implemented in the files in the folder `functions`.

The study is implemented as detailed in the paper. We initially wrote the code for the CEA algorithm simulation using Qiskit structures, but the slow transpilation on huge state vectors and lack of support for sparse matrices (https://github.com/Qiskit/qiskit/issues/12725) led to a change of the approach to a more conventional straight-forward application of the associated unitary matrices to obtain the relevant state vector. Currently, in the file `cea.py`, the CEA algorithm is implemented in the following manner:
- Initialize an m+n qubit register and apply the Hadamard gates to the first m qubits
- Obtain the statevector
- Generate a sparse unitary gate for the given Boolean vector function and apply it on the state just obtained
- Apply the (m+n)th Hadamard gates using a parallelized function based on `numba`
- The resulting vector corresponds to the result of the CEA before measurement, the measurement is performed using Pythons `choices` method, possibly with a custom seed for reproducibility
In the file `tests.py`, the original Qiskit-based approach can still be found as a metric for the correctness of this implementation.

The study itself stores the resulting masks using a custom binary file format. The functions `exp_empirical_study` and `get_exp_results` are the central functions here. The file format stores for each measurement a quadruple
`(round_index, input_mask, output_mask, walsh_transform, bias)`
with a space use of 14 bytes. Thus, a file with `r` measurement results to be stored is of a size of `14*r` bytes. Since the masks require at most `8` bits each for our current experiments, this is fully sufficient and yields very small file sizes, as observed.
Ultimately, the directory `experiment-results` hold the study results, and the directory `analysis-results` the associated confidence intervals. 