#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Naive GEMM, "bad" variant: threadIdx.x indexes the matrix ROW.
Consecutive threads in a warp (consecutive threadIdx.x) walk down a column of C
(stride N), so global loads/stores are NOT coalesced. Mirrors Boehm's kernel 1.
*/

__global__ void naive_gemm_bad(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C) {
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;
  if (row < M && col < N) {
    float sum = 0;
    for (int i = 0; i < K; ++i) {
      sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
  }
}
