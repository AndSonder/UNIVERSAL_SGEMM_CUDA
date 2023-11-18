#include "runner.cuh"
#include "matrix_utils.cuh"

#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int run_type = atoi(argv[4]);

    // Allocate memory for matrices
    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];
    // save reference result
    float* C_ref = new float[m * n];

    float* d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));
    cudaMalloc((void**)&d_C_ref, m * n * sizeof(float));

    // Initialize matrices: use range_init_matrix/randomize_matrix/zero_init_matrix
    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);
    zero_init_matrix(C, m * n);
    zero_init_matrix(C_ref, m * n);

    // Copy matrices to device
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ref, C_ref, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Run reference matrix multiplication
    run_cutlass_sgemm(A, B, C_ref, m, n, k);
    cudaDeviceSynchronize();

    // Run matrix multiplication
    switch (run_type) {
        case 0:
            run_sgemm_naive(A, B, C, m, n, k);
            break;
        default:
            std::cout << "Invalid run type" << std::endl;
            return 1;
    }
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref, d_C_ref, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Check result
    bool correct = true;
    float eps = 1e-6;
    for (int i = 0; i < m * n; i++) {
        if (abs(C[i] - C_ref[i]) > eps) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Result is correct!" << std::endl;
    } else {
        std::cout << "Result is incorrect!" << std::endl;
    }

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
}