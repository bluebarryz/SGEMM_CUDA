#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BLOCKSIZE>
__global__ void shared_mem_tiling_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
								  float beta, float *C) {

	const uint chunkRow = blockIdx.y;
	const uint chunkCol = blockIdx.x;

	__shared__ float AChunk[BLOCKSIZE * BLOCKSIZE];
	__shared__ float BChunk[BLOCKSIZE * BLOCKSIZE];

	const uint localRow = threadIdx.y;
	const uint localCol = threadIdx.x;

	A = &A[chunkRow * BLOCKSIZE * K];
	B = &B[chunkCol * BLOCKSIZE];
	C = &C[chunkRow * BLOCKSIZE * N + chunkCol * BLOCKSIZE];
	
	float sum = 0.0;
	for (int chunkIdx = 0; chunkIdx < K/BLOCKSIZE; ++chunkIdx) {
		AChunk[localRow * BLOCKSIZE + localCol] = A[localRow * K + localCol + BLOCKSIZE * chunkIdx];
		BChunk[localRow * BLOCKSIZE + localCol] = B[(localRow + BLOCKSIZE * chunkIdx) * N + localCol];
		__syncthreads();
		for (int kIdx = 0; kIdx < BLOCKSIZE; ++kIdx) {
			sum += AChunk[localRow * BLOCKSIZE + kIdx] *
					BChunk[kIdx * BLOCKSIZE + localCol];
		}
		__syncthreads();
	}
	C[localRow * N + localCol] = alpha * sum + beta * C[localRow * N + localCol];
}