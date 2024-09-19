# CUDA-Matrix-Multiplication-Project

/* 
Project: Matrix Multiplication  
This program compares the performance of CPU (serial) and GPU (parallel) implementations of matrix multiplication by tuning the dimensions of the input matrices.

The code includes the following key components:

1. **Include Directives**: 
   - Necessary libraries are included for time measurement, input/output operations, random number generation, and CUDA functions.

2. **GPU Kernel Function (`matmul`)**:
   - This function performs matrix multiplication on the GPU. 
   - Parameters:
     - `dim`: The size of each matrix (assuming square matrices).
     - `multiplier1` and `multiplier2`: Input matrices.
     - `product`: Output matrix where the result will be stored.
   - **Threading**: 
     - Each thread calculates a specific element in the resulting matrix based on its global thread ID.
     - The calculation accesses the appropriate rows and columns of the input matrices to compute the dot product.

3. **Timer Function (`timer`)**:
   - Measures the time taken for computations using `clock()`.
   - Outputs the elapsed time in seconds.

4. **Run Time Function (`run_time`)**:
   - Initializes the input matrices and fills them with random values.
   - Implements the CPU version of matrix multiplication and measures its execution time.
   - Allocates GPU memory for input and output matrices, copies the data from the host to the device, and launches the GPU kernel.
   - Synchronizes the device to ensure all computations are completed before copying the results back to the host.

5. **File Saving Function (`saveFile`)**:
   - Saves the runtime data from CPU and GPU calculations to text files for later analysis.

6. **Main Function**:
   - Defines a set of test matrix sizes.
   - Calls the `run_time` function for each size to perform the calculations and measure performance.
   - Saves the results to files and cleans up dynamically allocated memory.

This structure ensures that both CPU and GPU performances are compared effectively across different matrix dimensions, allowing for an assessment of the advantages of parallel computing in matrix operations.
