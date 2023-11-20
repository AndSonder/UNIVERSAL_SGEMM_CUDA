#pragma once

#include "kernels.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void run_sgemm_naive(const float *A, const float *B, float *C, int m, int n, int k) {
    dim3 block_size(32, 32);
    dim3 grid_size(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    sgemm_naive_kernel<<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_cutlass_sgemm(const float *A, const float *B, float *C, int m, int n, int k)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n);
    cublasDestroy(handle);
}

void run_sgemm_naive_cpu(const float *A, const float *B, float *C, int m, int n, int k) {
    sgemm_naive_cpu(A, B, C, m, n, k);
}

void run_sgemm_global_memory_coalescing(const float *A, const float *B, float *C, int m, int n, int k) {
    const int BLOCKSIZE = 32;
    dim3 block_size(32 * 32);
    dim3 grid_size(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    sgemm_global_memory_coalescing_kernel<BLOCKSIZE><<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_sgemm_shared_memory(const float *A, const float *B, float *C, int m, int n, int k) {
    const int BLOCKSIZE = 32;
    dim3 block_size(BLOCKSIZE * BLOCKSIZE);
    dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
    sgemm_shared_mem_kernel<BLOCKSIZE><<<grid_size, block_size>>>(A, B, C, m, n, k);
}