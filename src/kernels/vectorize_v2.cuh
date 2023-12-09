#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&pointer))[0]

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorize_v2_kernel(
    float *A,
    float *B,
    float *C,
    int M,
    int N,
    int K,
    int pitchA,
    int pitchB,
    int pitchC)
{
    const int thread_row = threadIdx.x / (BN / TN);
    const int thread_col = threadIdx.x % (BN / TN);

    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];
    float vector_outer_prod_A[2][TM] = {0.0};
    float vector_outer_prod_B[2][TN] = {0.0};
    float rst_each_thread[TM * TN] = {0.0};

    float ldg_reg_A[4] = {0.0};
    float ldg_reg_B[4] = {0.0};

    const int A_inner_row = threadIdx.x / (BK / 4);
    const int A_inner_col = threadIdx.x % (BK / 4);
    const int B_inner_row = threadIdx.x / (BN / 4);
    const int B_inner_col = threadIdx.x % (BN / 4);

    const int ldA = pitchA / sizeof(float);
    const int ldB = pitchB / sizeof(float);
    const int ldC = pitchC / sizeof(float);

    A += blockIdx.y * BM * ldA;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * ldB + blockIdx.x * BN;

    if (blockIdx.y * BM + A_inner_row < M)
        FETCH_FLOAT4(ldg_reg_A[0]) =
            FETCH_FLOAT4(A[OFFSET(A_inner_row, A_inner_col * 4, ldA)]);

    As[0][OFFSET(A_inner_col * 4 + 0, A_inner_row, BM)] = ldg_reg_A[0];
    As[0][OFFSET(A_inner_col * 4 + 1, A_inner_row, BM)] = ldg_reg_A[1];
    As[0][OFFSET(A_inner_col * 4 + 2, A_inner_row, BM)] = ldg_reg_A[2];
    As[0][OFFSET(A_inner_col * 4 + 3, A_inner_row, BM)] = ldg_reg_A[3];

    int dot_offset = 0 * BK;

    if (dot_offset + B_inner_row < K)
        FETCH_FLOAT4(ldg_reg_B[0]) =
            FETCH_FLOAT4(B[OFFSET(B_inner_row, B_inner_col * 4, ldB)]);

    FETCH_FLOAT4(Bs[0][OFFSET(B_inner_row, B_inner_col * 4, BN)]) =
        FETCH_FLOAT4(ldg_reg_B[0]);

    __syncthreads();

    int inner_index = 0;
    FETCH_FLOAT4(vector_outer_prod_A[0][0]) =
        FETCH_FLOAT4(As[0][OFFSET(inner_index, thread_row * TM + 0, BM)]);
    FETCH_FLOAT4(vector_outer_prod_A[0][4]) =
        FETCH_FLOAT4(As[0][OFFSET(inner_index, thread_row * TM + 4, BM)]);

    FETCH_FLOAT4(vector_outer_prod_B[0][0]) =
        FETCH_FLOAT4(Bs[0][OFFSET(inner_index, thread_col * TN / 2, BN)]);
    FETCH_FLOAT4(vector_outer_prod_B[0][4]) = FETCH_FLOAT4(
        Bs[0][OFFSET(inner_index, thread_col * TN / 2 + BN / 2, BN)]);

    int write_index = 1;
    int loadIndex;

    for (int dot_index = 1; dot_index <= ldA / BK; dot_index++)
    {
        int dot_offset = BK * dot_index;
        float ldg_reg_A[4] = {0.0};
        float ldg_reg_B[4] = {0.0};
        if (dot_index < ldA / BK)
        {
            if (blockIdx.y * BM + A_inner_row < M)
                FETCH_FLOAT4(ldg_reg_A[0]) = FETCH_FLOAT4(
                    A[OFFSET(A_inner_row, A_inner_col * 4 + dot_offset, ldA)]);

            if (dot_offset + B_inner_row < K)
                FETCH_FLOAT4(ldg_reg_B[0]) = FETCH_FLOAT4(
                    B[OFFSET(dot_offset + B_inner_row, B_inner_col * 4, ldB)]);
        }

        loadIndex = write_index ^ 1;

#pragma unroll
        for (int inner_index = 1; inner_index < BK; inner_index++)
        {
            FETCH_FLOAT4(vector_outer_prod_A[inner_index % 2][0]) = FETCH_FLOAT4(
                As[loadIndex][OFFSET(inner_index, thread_row * TM + 0, BM)]);
            FETCH_FLOAT4(vector_outer_prod_A[inner_index % 2][4]) = FETCH_FLOAT4(
                As[loadIndex][OFFSET(inner_index, thread_row * TM + 4, BM)]);

            FETCH_FLOAT4(vector_outer_prod_B[inner_index % 2][0]) = FETCH_FLOAT4(
                Bs[loadIndex][OFFSET(inner_index, thread_col * TN / 2, BN)]);
            FETCH_FLOAT4(vector_outer_prod_B[inner_index % 2][4]) =
                FETCH_FLOAT4(Bs[loadIndex][OFFSET(
                    inner_index, thread_col * TN / 2 + BN / 2, BN)]);

#pragma unroll
            for (int rst_each_thread_row = 0; rst_each_thread_row < TM; rst_each_thread_row++)
#pragma unroll
                for (int rst_each_thread_col = 0; rst_each_thread_col < TN;
                     rst_each_thread_col++)
                    rst_each_thread[OFFSET(rst_each_thread_row, rst_each_thread_col, TN)] +=
                        vector_outer_prod_A[(inner_index - 1) % 2][rst_each_thread_row] *
                        vector_outer_prod_B[(inner_index - 1) % 2][rst_each_thread_col];
        }

        if (dot_index < ldA / BK)
        {
            As[write_index][OFFSET(A_inner_col * 4 + 0, A_inner_row, BM)] =
                ldg_reg_A[0];
            As[write_index][OFFSET(A_inner_col * 4 + 1, A_inner_row, BM)] =
                ldg_reg_A[1];
            As[write_index][OFFSET(A_inner_col * 4 + 2, A_inner_row, BM)] =
                ldg_reg_A[2];
            As[write_index][OFFSET(A_inner_col * 4 + 3, A_inner_row, BM)] =
                ldg_reg_A[3];

            FETCH_FLOAT4(
                Bs[write_index][OFFSET(B_inner_row, B_inner_col * 4, BN)]) =
                FETCH_FLOAT4(ldg_reg_B[0]);
        }

        __syncthreads();

        FETCH_FLOAT4(vector_outer_prod_A[0][0]) = FETCH_FLOAT4(
            As[write_index][OFFSET(inner_index, thread_row * TM + 0, BM)]);
        FETCH_FLOAT4(vector_outer_prod_A[0][4]) = FETCH_FLOAT4(
            As[write_index][OFFSET(inner_index, thread_row * TM + 4, BM)]);

        FETCH_FLOAT4(vector_outer_prod_B[0][0]) = FETCH_FLOAT4(
            Bs[write_index][OFFSET(inner_index, thread_col * TN / 2, BN)]);
        FETCH_FLOAT4(vector_outer_prod_B[0][4]) =
            FETCH_FLOAT4(Bs[write_index][OFFSET(
                inner_index, thread_col * TN / 2 + BN / 2, BN)]);

#pragma unroll
        for (int rst_each_thread_row = 0; rst_each_thread_row < TM; rst_each_thread_row++)
#pragma unroll
            for (int rst_each_thread_col = 0; rst_each_thread_col < TN; rst_each_thread_col++)
                rst_each_thread[OFFSET(rst_each_thread_row, rst_each_thread_col, TN)] +=
                    vector_outer_prod_A[(BK - 1) % 2][rst_each_thread_row] *
                    vector_outer_prod_B[(BK - 1) % 2][rst_each_thread_col];
        write_index ^= 1;
    }

#pragma unroll
    for (int rst_each_thread_row = 0; rst_each_thread_row < TM; rst_each_thread_row++)
    {
        if (blockIdx.y * BM + thread_row * TM + rst_each_thread_row < M)
            FETCH_FLOAT4(C[OFFSET(thread_row * TM + rst_each_thread_row,
                                  thread_col * TN / 2, ldC)]) =
                FETCH_FLOAT4(rst_each_thread[OFFSET(rst_each_thread_row, 0, TN)]);
    }
#pragma unroll
    for (int rst_each_thread_row = 0; rst_each_thread_row < TM; rst_each_thread_row++)
    {
        if (blockIdx.y * BM + thread_row * TM + rst_each_thread_row < M)
            FETCH_FLOAT4(C[OFFSET(thread_row * TM + rst_each_thread_row,
                                  thread_col * TN / 2 + BN / 2, ldC)]) =
                FETCH_FLOAT4(rst_each_thread[OFFSET(rst_each_thread_row, 4, TN)]);
    }
}