#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B,
								   float beta, float *C) {
	assert(BM * BK == blockDim.x * blockDim.y);
	assert(BK * BN == blockDim.x * blockDim.y);

	const int chunkRow = blockIdx.y;
	const int chunkCol = blockIdx.x;
	const int localRowCompute = threadIdx.x / BN;
	const int localColCompute = threadIdx.x % BN;
	const int localRowLoadA = threadIdx.x / BK;
	const int localColLoadA = threadIdx.x % BK;
	const int localRowLoadB = threadIdx.x / BN;
	const int localColLoadB = threadIdx.x % BN;

	__shared__ float As[BM * BK];
	__shared__ float Bs[BK * BN];
	A = &A[chunkRow * BM * K];
	B = &B[chunkCol * BN];
	C = &C[chunkRow * BM * N + chunkCol * BN];
	float dotProds[TM] = {0.0};

	for (int chunkIdx = 0; chunkIdx < K / BK; ++chunkIdx) {
		As[localRowLoadA * BK + localColLoadA] =
			A[localRowLoadA * K + chunkIdx * BK + localColLoadA];
		Bs[localRowLoadB * BN + localColLoadB] =
			B[(chunkIdx * BK + localRowLoadB) * N + localColLoadB];
		__syncthreads();

		for (int j = 0; j < BK; ++j) {
			float BVal = Bs[j * BN + localColCompute];
			for (int i = 0; i < TM; ++i) {
				dotProds[i] += BVal * As[(localRowCompute * TM + i) * BK + j];
			}
		}
		__syncthreads();
	}

	for (int i = 0; i < TM; ++i) {
		C[(localRowCompute * TM + i) * N + localColCompute] =
			alpha * dotProds[i] + beta * C[(localRowCompute * TM + i) * N + localColCompute];
	}
}