#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

template <const int BLOCKSIZE>
__global__ void sgemm_global_memory_coalescing_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    if (x < M && y < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
        {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
}