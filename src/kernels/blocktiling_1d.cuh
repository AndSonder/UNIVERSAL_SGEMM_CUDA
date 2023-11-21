#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    // the output block that we want to compute in this threadblock
    const uint bx = blockIdx.y;
    const uint by = blockIdx.x;

    // allocate shared memory for the input and output submatrices
    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    // the inner row & col that we're accessing in this thread
    const uint thread_x = threadIdx.x % BN;
    const uint thread_y = threadIdx.x / BN;

    // advance pointers to the starting positions
    A += bx * BM * K;
    B += by * BN;
    C += bx * BM * N + by * BN;

    int global_x = bx * BM * K;
    int global_y = by * BN;

    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    const uint A_inner_x = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint A_inner_y = threadIdx.x / BK;
    const uint B_inner_x = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint B_inner_y = threadIdx.x / BN;

    // allocate thread-local cache for results in registerfile
    float thread_results[TM] = {0.0};

    // outer loop over block tiles
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        // load the next block of the input matrices into shared memory
        A_shared[A_inner_y * BK + A_inner_x] = (global_x + A_inner_y * K + A_inner_x < M * K) ? A[A_inner_y * K + A_inner_x] : 0.0f;
        B_shared[B_inner_y * BN + B_inner_x] = (global_y + B_inner_y * N + B_inner_x < N * K) ? B[B_inner_y * N + B_inner_x] : 0.0f;

        // wait for all threads to finish loading
        __syncthreads();

        // advance the pointers
        A += BK;
        B += BK * N;
        global_x += BK;
        global_y += BK * N;

        // compute the partial sum
        for (uint dot_idx = 0; dot_idx < BK; dot_idx++)
        {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            float tmp_b = B_shared[dot_idx * BN + thread_x];
            for (uint res_idx = 0; res_idx < TM; res_idx++)
            {
                // (threadRow * TM + resIdx) * BK + dotIdx
                thread_results[res_idx] += A_shared[(thread_y * TM + res_idx) * BK + dot_idx] * tmp_b;
            }
        }

        // wait for all threads to finish computing
        __syncthreads();
    }

    for (uint res_idx = 0; res_idx < TM; res_idx++)
    {
        if (bx * BM + thread_y * TM + res_idx < M && by * BN + thread_x < N)
        {
            C[(thread_y * TM + res_idx) * N + thread_x] = thread_results[res_idx];
        }
    }
}