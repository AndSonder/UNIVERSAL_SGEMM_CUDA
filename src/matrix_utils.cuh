#pragma once

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>

void range_init_matrix(float *mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = i;
    }
}

void randomize_matrix(float *mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = rand() % 100;
    }
}

void zero_init_matrix(float *mat, int N) {
    memset(mat, 0, N * sizeof(float));
}

void print_matrix(const float *A, int M, int N){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++){
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}