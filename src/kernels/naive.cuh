#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

void sgemm_naive_cpu(float *A, float *B, float *C, int M, int N, int K)
{
    for (int x = 0; x < M; x ++) {
        for (int y = 0; y < N; y ++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i ++) {
                sum += A[x * K + i] * B[i * N + y];
            }
            C[x * N + y] = sum;
        }
    }
}

__global__ void sgemm_naive_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) 
    {
        float sum = 0.0f;
        for (int i = 0; i < K;i ++) {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
 }