#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Naive GEMM, "better" variant: threadIdx.x indexes the matrix COL.
Consecutive threads in a warp (consecutive threadIdx.x) now walk across a row of
C (stride 1) and read the same row of A, so global accesses coalesce. This is
the row/col swap from Boehm's kernel 1 -> kernel 2 (GMEM coalescing).
*/

__global__ void naive_gemm_better(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  if (row < M && col < N) {
    float sum = 0;
    for (int i = 0; i < K; ++i) {
      sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
  }
}
