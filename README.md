# GPU-Accelerated Prime Number Generator

## Overview

This is a C++ program that calculates prime numbers in the most efficient way possible. I've implemented a segmented Sieve of Eratosthenes algorithm, offloading to the GPU using CUDA. It also employs multi-threading for handling writing the primes to a file.

## Functionality

The program performs the following steps:

1.  **Command-Line Argument Parsing:** Takes a single command-line argument representing the upper limit up to which prime numbers should be found.
2.  **Base Prime Generation (CPU):** Calculates all prime numbers up to the square root of the upper limit using a standard Sieve of Eratosthenes algorithm implemented on the CPU. These "base primes" are used to sieve larger segments on the GPU.
3.  **GPU-Accelerated Segmented Sieving:**
    * Divides the range of numbers (from the square root limit + 1 up to the upper limit) into smaller segments.
    * For each segment, it transfers the segment data and the pre-calculated base primes to the GPU.
    * A CUDA kernel function performs the sieving process on the GPU, marking non-prime numbers within the segment.
    * The results are transferred back to the CPU.
4.  **Prime Collection and Storage:** Prime numbers identified in each segment on the GPU are collected and stored in a buffer.
5.  **Multi-threaded Writing:** A separate writer thread continuously monitors the buffer of found primes and writes them to a file named `primes.txt`. This happens concurrently with the GPU processing.
6.  **Performance Measurement:** The program measures and reports the time taken for different stages of the process, including the CPU sieve, GPU processing, and overall execution time.
7.  **Final Summary:** After processing all segments, the program prints a summary including the total number of primes found and the largest prime number within the specified limit.

## Usage


### Prerequisites 

* **CUDA-enabled NVIDIA GPU:** This program requires an NVIDIA GPU with CUDA support.
* **NVIDIA CUDA Toolkit:** You need to have the NVIDIA CUDA Toolkit installed on your system. Ensure that the `nvcc` compiler is in your system's PATH.

### Running the Program

1.  Open a terminal or command prompt in the executable's directory
2. Run it, passing the upper limit as a parameter to the command:
```bash
list_prime```


### Compilation

The compiled program is already provided as `listprimes.exe`

If you want to make changes yourself and recompile:
1.  Save the  code as a `.cu` file.
2.  Compile the code using the `nvcc` compiler:

    ```bash
    nvcc prime_sieve.cu -o prime_sieve.exe
    ```


