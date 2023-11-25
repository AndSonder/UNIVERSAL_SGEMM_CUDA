#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&pointer))[0]

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorize_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    uint global_a_row = cRow * BM;
    uint global_a_col = 0;
    uint global_b_row = 0;
    uint global_b_col = cCol * BN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint A_inner_row = threadIdx.x / (BK / 4);
    const uint A_inner_col = threadIdx.x % (BK / 4);
    const uint B_inner_row = threadIdx.x / (BN / 4);
    const uint B_inner_col = threadIdx.x % (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    float reg_m[TM] = {0.0};
    float reg_n[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        As[(A_inner_col * 4 + 0) * BM + A_inner_row] =
            (global_a_row + A_inner_row < M && global_a_col + A_inner_col * 4 < K) ? A[A_inner_row * K + A_inner_col * 4] : 0.0f;
        As[(A_inner_col * 4 + 1) * BM + A_inner_row] =
            (global_a_row + A_inner_row < M && global_a_col + A_inner_col * 4 + 1 < K) ? A[A_inner_row * K + A_inner_col * 4 + 1] : 0.0f;
        As[(A_inner_col * 4 + 2) * BM + A_inner_row] =
            (global_a_row + A_inner_row < M && global_a_col + A_inner_col * 4 + 2 < K) ? A[A_inner_row * K + A_inner_col * 4 + 2] : 0.0f;
        As[(A_inner_col * 4 + 3) * BM + A_inner_row] =
            (global_a_row + A_inner_row < M && global_a_col + A_inner_col * 4 + 3 < K) ? A[A_inner_row * K + A_inner_col * 4 + 3] : 0.0f;

        Bs[B_inner_row * BN + B_inner_col * 4] = (global_b_row + B_inner_row < K && global_b_col + B_inner_col * 4 < N) ? B[B_inner_row * N + B_inner_col * 4] : 0.0f;
        Bs[B_inner_row * BN + B_inner_col * 4 + 1] = (global_b_row + B_inner_row < K && global_b_col + B_inner_col * 4 + 1 < N) ? B[B_inner_row * N + B_inner_col * 4 + 1] : 0.0f;
        Bs[B_inner_row * BN + B_inner_col * 4 + 2] = (global_b_row + B_inner_row < K && global_b_col + B_inner_col * 4 + 2 < N) ? B[B_inner_row * N + B_inner_col * 4 + 2] : 0.0f;
        Bs[B_inner_row * BN + B_inner_col * 4 + 3] = (global_b_row + B_inner_row < K && global_b_col + B_inner_col * 4 + 3 < N) ? B[B_inner_row * N + B_inner_col * 4 + 3] : 0.0f;

        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        global_a_col += BK;
        global_b_row += BK;

        // calculate per-thread results
#pragma unroll
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
#pragma unroll
            // block into registers
            for (uint i = 0; i < TM; ++i)
            {
                reg_m[i] = As[dotIdx * BM + threadRow * TM + i];
            }
#pragma unroll
            for (uint i = 0; i < TN; ++i)
            {
                reg_n[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
#pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
            {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                {
                    threadResults[resIdxM * TN + resIdxN] +=
                        reg_m[resIdxM] * reg_n[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1)
    {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4)
        {
            for (int i = 0; i < 4; i++)
            {
                // judge row < m and col < n
                if (cRow * BM + threadRow * TM + resIdxM < M && cCol * BN + threadCol * TN + resIdxN + i < N)
                {
                    C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + i] = threadResults[resIdxM * TN + resIdxN + i];
                }
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorize_float4_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    uint global_a_row = cRow * BM;
    uint global_a_col = 0;
    uint global_b_row = 0;
    uint global_b_col = cCol * BN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint A_inner_row = threadIdx.x / (BK / 4);
    const uint A_inner_col = threadIdx.x % (BK / 4);
    const uint B_inner_row = threadIdx.x / (BN / 4);
    const uint B_inner_col = threadIdx.x % (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    float reg_m[TM] = {0.0};
    float reg_n[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // populate the SMEM caches
        // transpose A while loading it
        float4 tmp =
            reinterpret_cast<float4 *>(&A[A_inner_row * K + A_inner_col * 4])[0];

        As[(A_inner_col * 4 + 0) * BM + A_inner_row] =
            (global_a_row + A_inner_row < M && global_a_col + A_inner_col * 4 < K) ? tmp.x : 0.0f;
        As[(A_inner_col * 4 + 1) * BM + A_inner_row] =
            (global_a_row + A_inner_row < M && global_a_col + A_inner_col * 4 + 1 < K) ? tmp.y : 0.0f;
        As[(A_inner_col * 4 + 2) * BM + A_inner_row] =
            (global_a_row + A_inner_row < M && global_a_col + A_inner_col * 4 + 2 < K) ? tmp.z : 0.0f;
        As[(A_inner_col * 4 + 3) * BM + A_inner_row] =
            (global_a_row + A_inner_row < M && global_a_col + A_inner_col * 4 + 3 < K) ? tmp.w : 0.0f;

        float4 tmp2 = reinterpret_cast<float4 *>(&B[B_inner_row * N + B_inner_col * 4])[0];
        Bs[B_inner_row * BN + B_inner_col * 4] = (global_b_row + B_inner_row < K && global_b_col + B_inner_col * 4 < N) ? tmp2.x : 0.0f;
        Bs[B_inner_row * BN + B_inner_col * 4 + 1] = (global_b_row + B_inner_row < K && global_b_col + B_inner_col * 4 + 1 < N) ? tmp2.y : 0.0f;
        Bs[B_inner_row * BN + B_inner_col * 4 + 2] = (global_b_row + B_inner_row < K && global_b_col + B_inner_col * 4 + 2 < N) ? tmp2.z : 0.0f;
        Bs[B_inner_row * BN + B_inner_col * 4 + 3] = (global_b_row + B_inner_row < K && global_b_col + B_inner_col * 4 + 3 < N) ? tmp2.w : 0.0f;

        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        global_a_col += BK;
        global_b_row += BK;

        // calculate per-thread results
#pragma unroll
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
#pragma unroll
            // block into registers
            for (uint i = 0; i < TM; ++i)
            {
                reg_m[i] = As[dotIdx * BM + threadRow * TM + i];
            }
#pragma unroll
            for (uint i = 0; i < TN; ++i)
            {
                reg_n[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
#pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
            {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                {
                    threadResults[resIdxM * TN + resIdxN] +=
                        reg_m[resIdxM] * reg_n[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1)
    {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4)
        {
            for (int i = 0; i < 4; i++)
            {
                if (cRow * BM + threadRow * TM + resIdxM < M && cCol * BN + threadCol * TN + resIdxN + i < N)
                {
                    C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + i] = threadResults[resIdxM * TN + resIdxN + i];
                }
            }
        }
    }
}