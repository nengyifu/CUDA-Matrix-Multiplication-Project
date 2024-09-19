#include <time.h>
#include <ctime>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <random>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

/* GPU kernel function
dim: the number of rows in multiplier1 and the number of columns in multiplier2
m: the number of columns in multiplier1 and the number of rows in multiplier2
multiplier1: input matrix of size n*m
multiplier2: input matrix of size m*n
product: output matrix of size n*n
*/

// Each thread computes one element of the output matrix
__global__ void matmul(int n, int m, double* multiplier1, double* multiplier2, double* product) {
    unsigned int threadNum = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID

    // Calculate the row and column index for the output matrix
    unsigned int row = threadNum / n;
    unsigned int col = threadNum % n;

    // Initialize the product value
    if (row < n && col < n) {
        double value = 0.0;
        for (int i = 0; i < m; ++i) {
            value += multiplier1[row * m + i] * multiplier2[i * n + col];
        }
        product[threadNum] = value;
    }
}

double timer(clock_t t_begin) {
    double time_elapsed;
    clock_t t_end = clock();
    time_elapsed = double(t_end - t_begin) / CLOCKS_PER_SEC;
    printf("Time elapsed: %e s\n", time_elapsed);
    return time_elapsed;
}

void run_time(unsigned short n, float* runtime_CPU, float* runtime_GPU) {
    clock_t t0;
    int m = 3 * n; // Set m to be double the value of n

    // Allocate CPU memory for matrices
    double* x = (double*)malloc(n * m * sizeof(double)); // n*m matrix
    double* y = (double*)malloc(m * n * sizeof(double)); // m*n matrix
    srand(time(0)); // Set the random seed

    // Initialize the input matrices with random values
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < m; k++) {
            x[j * m + k] = (rand() + 1.0) / (rand() + 1.0);
        }
    }
    for (int j = 0; j < m; j++) {
        for (int k = 0; k < n; k++) {
            y[j * n + k] = (rand() + 1.0) / (rand() + 1.0);
        }
    }

    // Implement a CPU version of matrix multiplication
    t0 = clock();
    double* w = (double*)malloc(n * n * sizeof(double)); // Output matrix of size n*n
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
            double we = 0.0;
            for (int m_idx = 0; m_idx < m; m_idx++) {
                we += x[j * m + m_idx] * y[m_idx * n + k];
            }
            w[j * n + k] = we;
        }
    }
    runtime_CPU[0] = timer(t0);
    free(w);

    // GPU version
    double* cudaX, * cudaY, * cudaZ;
    cudaMalloc(&cudaX, n * m * sizeof(double)); // Allocate GPU memory for x
    cudaMalloc(&cudaY, m * n * sizeof(double)); // Allocate GPU memory for y
    cudaMalloc(&cudaZ, n * n * sizeof(double)); // Allocate GPU memory for product

    // Copy input matrices x and y from CPU to GPU
    cudaMemcpy(cudaX, x, n * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaY, y, m * n * sizeof(double), cudaMemcpyHostToDevice);
    free(x);
    free(y);

    // Determine block and grid sizes
    int numThreads = 1024; // Maximum number of threads per block
    int numBlocks = (n * n + numThreads - 1) / numThreads; // Calculate number of blocks needed

    t0 = clock();
    matmul << <numBlocks, numThreads >> > (n, m, cudaX, cudaY, cudaZ); // Launch the kernel
    cudaDeviceSynchronize(); // Wait until GPU is done
    runtime_GPU[0] = timer(t0);

    // Copy results from GPU to CPU
    double* z = (double*)malloc(n * n * sizeof(double));
    cudaMemcpy(z, cudaZ, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory resources
    cudaFree(cudaX);
    cudaFree(cudaY);
    cudaFree(cudaZ);
    free(z);
}

void saveFile(string name, float* data, unsigned int x) {
    ofstream write;
    string tpStr = ".txt";
    write.open(name + tpStr);

    for (auto i = 0; i < x; ++i) {
        write << data[i] << "	";
    }
    write.close();
}

int main() {
    const unsigned short no_test = 4;
    unsigned short trials[no_test] = {222, 333,444, 555};
    unsigned short n;
    float* runtime_CPU = new float[no_test];
    float* runtime_GPU = new float[no_test];

    for (auto i = 0; i < no_test; ++i) {
        n = trials[i];
        run_time(n, runtime_CPU + i, runtime_GPU + i);
        cout << endl;
    }

    saveFile("runtime_CPU", runtime_CPU, no_test);
    saveFile("runtime_GPU", runtime_GPU, no_test);
    delete[] runtime_CPU;
    delete[] runtime_GPU;
    return 0;
}
