// This file contain the kernels for the CUDADL library

#include "../include/kernels.h"

////////////////// Matmul functions /////////////////////

// Simple stupid matrix multiplication on the GPU
template <typename T>
__global__ void matmul_kernel_v00(T *A, T *B, T *C, int M, int N, int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < M && j < N) {
    C[i * N + j] = 0;
    for (int k = 0; k < K; k++) {
      C[i * N + j] += A[i * K + k] * B[k * N + j];
    }
  }
}

// Smarter multiplication on the GPU using shared memory
template <typename T>
__global__ void matmul_kernel_v01(T *A, T *B, T *C, int M, int N, int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int local_i = threadIdx.x;
  int local_j = threadIdx.y;

  __shared__ T shared_a[TPB * TPB];
  __shared__ T shared_b[TPB * TPB];

  int acc = 0;
  for (int k = 0; k < K; k += TPB) {
    if (i < M && k + local_j < K) {
      shared_a[local_i * TPB + local_j] = A[i * K + k + local_j];
    }
    if (j < K && k + local_i < N) {
      shared_b[local_i * TPB + local_j] = B[(k + local_i) * N + j];
    }
    __syncthreads();

    for (int p = 0; p < min(TPB, K - k); p++) {
      if (i < M && j < N) {
        acc += shared_a[local_i * TPB + p] * shared_b[p * TPB + local_j];
      }
    }
  }

  if (i < M && j < N) {
    C[i * N + j] = acc;
  }
}

// Launch kernel on the GPU
template <typename T>
void launch_matmul(T *A, T *B, T *C, int M, int N, int K, int version) {
  dim3 block(TPB, TPB);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  if (version == 0) {
    matmul_kernel_v00<T><<<grid, block>>>(A, B, C, M, N, K);
  } else if (version == 1) {
    matmul_kernel_v01<T><<<grid, block>>>(A, B, C, M, N, K);
  }
}

////////////////// Initialization functions //////////////////////////
// Initialize the Matrixes on the GPU
template <typename T> __global__ void init_mat(T *A, int M, int N, int value) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < M && j < N) {
    A[i * N + j] = value;
  }
}

template <typename T> void init_matrix(T *A, int M, int N, int value) {
  dim3 block(TPB, TPB);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  init_mat<T><<<grid, block>>>(A, M, N, value);
}
