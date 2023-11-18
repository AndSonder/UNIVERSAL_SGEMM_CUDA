#pragma once

#include <cuda_runtime.h>


__global__ void sgemm_naive_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) 
    {
        float sum = 0.0f;
        for (int i = 0; i < K;i ++) {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
 }