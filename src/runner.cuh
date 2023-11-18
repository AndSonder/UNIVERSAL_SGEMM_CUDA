#pragma once

#include "kernels.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cublas_v2.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

void run_sgemm_naive(const float *A, const float *B, float *C, int m, int n, int k) {
    dim3 block_size(32, 32);
    dim3 grid_size(CEIL_DIV(n, block_size.x), CEIL_DIV(m, block_size.y));
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
