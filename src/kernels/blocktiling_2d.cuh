#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm_blocktiling_2d_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    // the output block that we want to compute in this threadblock
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    // // A thread is responsible for calculating TM*TN elements in the blocktile
    const uint num_threads_block_tile = (BM * BN) / (TM * TN);

    // allocate shared memory for the input and output submatrices
    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    // the inner row & col that we're accessing in this thread
    const uint thread_row = threadIdx.x / (BN / TN);
    const uint thread_col = threadIdx.x % (BN / TN);

    // advance pointers to the starting positions
    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    // use to avoid out-of-bounds accesses
    int global_m_pos = c_row * BM * K;
    int global_n_pos = c_col * BN;
    const uint m_size = M * K;
    const uint n_size = N * K;

    assert((BM * BN) / (TM * TN) == blockDim.x);

    const uint A_inner_row = threadIdx.x / BK; // warp-level GMEM coalescing
    const uint A_inner_col = threadIdx.x % BK;
    const uint stride_a = num_threads_block_tile / BK;
    const uint B_inner_row = threadIdx.x / BN; // warp-level GMEM coalescing
    const uint B_inner_col = threadIdx.x % BN;
    const uint stride_b = num_threads_block_tile / BN;

    // allocate thread-local cache for results in registerfile
    float thread_results[TM * TN] = {0.0};
    float reg_m[TM] = {0.0};
    float reg_n[TN] = {0.0};

    // outer loop over block tiles
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        // load the next block of the input matrices into shared memory
        for (uint load_offset = 0; load_offset < BM; load_offset += stride_a)
        {
            A_shared[(A_inner_row + load_offset) * BK + A_inner_col] = (global_m_pos + (A_inner_row + load_offset) * K + A_inner_col < m_size) ? A[(A_inner_row + load_offset) * K + A_inner_col] : 0.0f;
        }
        for (uint load_offset = 0; load_offset < BK; load_offset += stride_b)
        {
            B_shared[(B_inner_row + load_offset) * BN + B_inner_col] = (global_n_pos + (B_inner_row + load_offset) * N + B_inner_col < n_size) ? B[(B_inner_row + load_offset) * N + B_inner_col] : 0.0f;
        }

        // wait for all threads to finish loading
        __syncthreads();

        // advance the pointers
        A += BK;
        B += BK * N;
        global_m_pos += BK;
        global_n_pos += BK * N;

        // compute the partial sum
        for (uint dot_idx = 0; dot_idx < BK; dot_idx++)
        {
            // load relevant As & Bs entries into registers
            for (uint i = 0; i < TM; i++)
            {
                reg_m[i] = A_shared[(thread_row * TM + i) * BK + dot_idx];
            }
            for (uint i = 0; i < TN; i++)
            {
                reg_n[i] = B_shared[dot_idx * BN + thread_col * TN + i];
            }

            // perform outer product on register cache, accumulate
            // into threadResults
            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m++)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n++)
                {
                    thread_results[res_idx_m * TN + res_idx_n] += reg_m[res_idx_m] * reg_n[res_idx_n];
                }
            }
        }

        // wait for all threads to finish computing
        __syncthreads();
    }

    for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m++)
    {
        for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n++)
        {
            if (c_row * BM + thread_row * TM + res_idx_m < M && c_col * BN + thread_col * TN + res_idx_n < N)
            {
                C[(thread_row * TM + res_idx_m) * N + thread_col * TN + res_idx_n] = thread_results[res_idx_m * TN + res_idx_n];
            }
        }
    }
}