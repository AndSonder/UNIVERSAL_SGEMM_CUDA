#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    // the output block that we want to compute in this threadblock
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;

    // allocate shared memory for the input and output submatrices
    __shared__ float A_shared[BLOCKSIZE * BLOCKSIZE];
    __shared__ float B_shared[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const uint thread_x = threadIdx.x % BLOCKSIZE;
    const uint thread_y = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting positions
    A += bx * BLOCKSIZE * K;
    B += by * BLOCKSIZE;
    C += bx * BLOCKSIZE * N + by * BLOCKSIZE;

    float tmp = 0.0f;
    for (int i = 0;i < K;i += BLOCKSIZE)
    {
        // load the next block of the input matrices into shared memory
        A_shared[thread_x * BLOCKSIZE + thread_y] = A[thread_x * K + thread_y];
        B_shared[thread_x * BLOCKSIZE + thread_y] = B[thread_x * N + thread_y];

        // wait for all threads to finish loading
        __syncthreads();

        // advance the pointers
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        
        // compute the partial sum
        for (int j = 0; j < BLOCKSIZE; j++)
        {
            tmp += A_shared[thread_x * BLOCKSIZE + j] * B_shared[j * BLOCKSIZE + thread_y];
        }

        // wait for all threads to finish computing
        __syncthreads();
    }
    if (bx * BLOCKSIZE + thread_x < M && by * BLOCKSIZE + thread_y < N) {
        C[thread_x * N + thread_y] = tmp;
    }
}