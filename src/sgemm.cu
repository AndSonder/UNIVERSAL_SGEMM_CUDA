#include "runner.cuh"
#include "matrix_utils.cuh"

#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#define KernelErrChk()                                                        \
    {                                                                         \
        cudaError_t errSync = cudaGetLastError();                             \
        cudaError_t errAsync = cudaDeviceSynchronize();                       \
        if (errSync != cudaSuccess)                                           \
        {                                                                     \
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
        if (errAsync != cudaSuccess)                                          \
        {                                                                     \
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync)); \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

bool run_kernel(float *A,
                float *B,
                float *C,
                int m,
                int n,
                int k,
                size_t pitch_A = 0,
                size_t pitch_B = 0,
                size_t pitch_C = 0,
                int run_type = 0)
{
    switch (run_type)
    {
    case 0:
        run_sgemm_naive(A, B, C, m, n, k);
        return true;
    case 1:
        run_sgemm_global_memory_coalescing(A, B, C, m, n, k);
        return true;
    case 2:
        run_sgemm_shared_memory(A, B, C, m, n, k);
        return true;
    case 3:
        run_sgemm_blocktiling_1d(A, B, C, m, n, k);
        return true;
    case 4:
        run_sgemm_blocktiling_2d(A, B, C, m, n, k);
        return true;
    case 5:
        run_sgemm_vectorize(A, B, C, m, n, k);
        return true;
    case 6:
        run_sgemm_vectorize_v2(A, B, C, m, n, k, pitch_A, pitch_B, pitch_C);
        return true;
    default:
        printf("Invalid run type\n");
        return false;
    }
}

int main(int argc, char **argv)
{
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int run_type = atoi(argv[4]);

    // Allocate memory for matrices
    float *A, *B, *C, *C_ref;
    float *d_A, *d_B, *d_C, *d_C_ref;
    // for pitch version
    float *d_A_p, *d_B_p, *d_C_p;
    size_t pitch_A, pitch_B, pitch_C;

    A = new float[m * k];
    B = new float[k * n];
    C = new float[m * n];
    // save reference result
    C_ref = new float[m * n];

    // Initialize matrices: use range_init_matrix/randomize_matrix/zero_init_matrix
    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);
    zero_init_matrix(C, m * n);
    zero_init_matrix(C_ref, m * n);

    cudaMalloc((void **)&d_A, m * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * n * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));
    cudaMalloc((void **)&d_C_ref, m * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ref, C_ref, m * n * sizeof(float), cudaMemcpyHostToDevice);

    if (run_type >= 6)
    {
        // use cudaMallocPitch to allocate memory for matrices
        cudaMallocPitch((void **)&d_A_p, &pitch_A, k * sizeof(float), m);
        cudaMallocPitch((void **)&d_B_p, &pitch_B, n * sizeof(float), k);
        cudaMallocPitch((void **)&d_C_p, &pitch_C, n * sizeof(float), m);
        // Copy matrices to device
        cudaMemcpy2D(d_A_p, pitch_A, A, k * sizeof(float), k * sizeof(float), m, cudaMemcpyHostToDevice);
        cudaMemcpy2D(d_B_p, pitch_B, B, n * sizeof(float), n * sizeof(float), k, cudaMemcpyHostToDevice);
        cudaMemcpy2D(d_C_p, pitch_C, C, n * sizeof(float), n * sizeof(float), m, cudaMemcpyHostToDevice);
    }

    // Run reference matrix multiplication
    run_cutlass_sgemm(d_A, d_B, d_C_ref, m, n, k);

    cudaDeviceSynchronize();
    KernelErrChk();

    // Run matrix multiplication
    bool run_success = false;
    if (run_type >= 6)
    {
        run_success =
            run_kernel(d_A_p, d_B_p, d_C_p, m, n, k, pitch_A, pitch_B, pitch_C, run_type);
    }
    else
    {
        run_success =
            run_kernel(d_A, d_B, d_C, m, n, k, pitch_A, pitch_B, pitch_C, run_type);
    }

    if (!run_success)
    {
        std::cout << "Invalid run type" << std::endl;
        return 0;
    }
    cudaDeviceSynchronize();
    KernelErrChk();

    // Copy result back to host
    if (run_type >= 6)
    {
        cudaMemcpy2D(
            C,
            n * sizeof(float),
            d_C_p,
            pitch_C,
            n * sizeof(float),
            m,
            cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(C_ref, d_C_ref, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Check result
    bool correct = true;
    float eps = 1e-6;
    for (int i = 0; i < m * n; i++)
    {
        if (abs(C[i] - C_ref[i]) > eps)
        {
            printf("Error at position %d, expected %f, get %f\n", i, C_ref[i], C[i]);
            correct = false;
            break;
        }
    }

    if (correct)
    {
        // run speed test
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            if (run_type >= 6)
            {
                run_kernel(d_A_p, d_B_p, d_C_p, m, n, k, pitch_A, pitch_B, pitch_C, run_type);
            }
            else 
            {
                run_kernel(d_A, d_B, d_C, m, n, k, pitch_A, pitch_B, pitch_C, run_type);
            }
            cudaDeviceSynchronize();
        }
        KernelErrChk();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsed_time = 0.0;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        float avg_run_time = elapsed_time * 1000 / 100;
        std::cout << "Average run time: " << avg_run_time << " us" << std::endl;
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