#pragma once

#include "kernels.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void run_sgemm_naive(float *A, float *B, float *C, int m, int n, int k)
{
    dim3 block_size(32, 32);
    dim3 grid_size(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    sgemm_naive_kernel<<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_cutlass_sgemm(float *A, float *B, float *C, int m, int n, int k)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n);
    cublasDestroy(handle);
}

void run_sgemm_naive_cpu(float *A, float *B, float *C, int m, int n, int k)
{
    sgemm_naive_cpu(A, B, C, m, n, k);
}

void run_sgemm_global_memory_coalescing(float *A, float *B, float *C, int m, int n, int k)
{
    const int BLOCKSIZE = 32;
    dim3 block_size(32 * 32);
    dim3 grid_size(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    sgemm_global_memory_coalescing_kernel<BLOCKSIZE><<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_sgemm_shared_memory(float *A, float *B, float *C, int m, int n, int k)
{
    const int BLOCKSIZE = 32;
    dim3 block_size(BLOCKSIZE * BLOCKSIZE);
    dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
    sgemm_shared_mem_kernel<BLOCKSIZE><<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_sgemm_blocktiling_1d(float *A, float *B, float *C, int m, int n, int k)
{
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 grid_size(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 block_size((BM * BN) / TM);
    sgemm_blocktiling_1d_kernel<BM, BN, BK, TM>
        <<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void run_sgemm_blocktiling_2d(float *A, float *B, float *C, int m, int n, int k)
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    if (m >= 128 && n >= 128)
    {
        const uint BM = 128;
        const uint BN = 128;
        dim3 grid_size(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
        dim3 block_size((BM * BN) / (TM * TN));
        sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN>
            <<<grid_size, block_size>>>(A, B, C, m, n, k);
    }
    else
    {
        const uint BM = 64;
        const uint BN = 64;
        dim3 grid_size(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
        dim3 block_size((BM * BN) / (TM * TN));
        sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN>
            <<<grid_size, block_size>>>(A, B, C, m, n, k);
    }
}

void run_sgemm_vectorize(float *A, float *B, float *C, int m, int n, int k)
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    const uint BM = 128;
    const uint BN = 128;
    dim3 grid_size(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 block_size((BM * BN) / (TM * TN));
    if (m % 4 == 0 && n % 4 == 0 && k % 4 == 0)
    {
        sgemm_vectorize_float4_kernel<BM, BN, BK, TM, TN>
            <<<grid_size, block_size>>>(A, B, C, m, n, k);
    }
    else
    {
        sgemm_vectorize_kernel<BM, BN, BK, TM, TN>
            <<<grid_size, block_size>>>(A, B, C, m, n, k);
    }
}

void run_sgemm_vectorize_v2(float *A,
                            float *B,
                            float *C,
                            int m,
                            int n,
                            int k,
                            size_t pitch_A,
                            size_t pitch_B,
                            size_t pitch_C)
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    const uint BM = 128;
    const uint BN = 128;
    dim3 grid_size(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 block_size((BM * BN) / (TM * TN));
    sgemm_vectorize_v2_kernel<BM, BN, BK, TM, TN>
        <<<grid_size, block_size>>>(A, B, C, m, n, k, pitch_A, pitch_B, pitch_C);
}
