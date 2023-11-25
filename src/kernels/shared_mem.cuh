#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    // the output block that we want to compute in this threadblock
    const uint c_row = blockIdx.x;
    const uint c_col = blockIdx.y;

    // allocate shared memory for the input and output submatrices
    __shared__ float A_shared[BLOCKSIZE * BLOCKSIZE];
    __shared__ float B_shared[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const uint thread_row = threadIdx.x / BLOCKSIZE;
    const uint thread_col = threadIdx.x % BLOCKSIZE;

    // advance pointers to the starting positions
    A += c_row * BLOCKSIZE * K;
    B += c_col * BLOCKSIZE;
    C += c_row * BLOCKSIZE * N + c_col * BLOCKSIZE;

    float tmp = 0.0f;
    for (int i = 0;i < K;i += BLOCKSIZE)
    {
        // load the next block of the input matrices into shared memory
        A_shared[thread_col * BLOCKSIZE + thread_row] = (c_row * BLOCKSIZE + thread_col < M && i + thread_row < K) ? A[thread_col * K + thread_row] : 0.0f;
        B_shared[thread_col * BLOCKSIZE + thread_row] = (c_col * BLOCKSIZE + thread_row < N && i + thread_col < K) ? B[thread_col * N + thread_row] : 0.0f;

        // wait for all threads to finish loading
        __syncthreads();

        // compute the partial sum
        for (int j = 0; j < BLOCKSIZE; j++)
        {
            tmp += A_shared[thread_col * BLOCKSIZE + j] * B_shared[j * BLOCKSIZE + thread_row];
        }

        // wait for all threads to finish computing
        __syncthreads();

        // advance the pointers
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    if (c_row * BLOCKSIZE + thread_col < M && c_col * BLOCKSIZE + thread_row < N) {
        C[thread_col * N + thread_row] = tmp;
    }
}