// This file is used to test the CUDADL library
// Author: Miolo

#include <iostream>

#define TPB 32
#define DEBUG false // Print the matrices

/////////////////// Multiplication function/////////////////

// We start by writing a simple matrix multiplication running on the CPU
// We have A: MxK, B: KxN, C: MxN
template <typename T> void matrix_mul(T *A, T *B, T *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0;
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

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

/////////////////// Init and Verification /////////////////

// Initialize a matrix with random values
template <typename T> void init(T *A, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = rand() % 10;
    }
  }
}

// Initialize the Matrixes on the GPU
template <typename T> __global__ void init_kernel(T *A, int M, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < M && j < N) {
    A[i * N + j] = i + j;
  }
}

// Verify that two matrixes are the same
template <typename T> bool verify(T *A, T *B, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (A[i * N + j] != B[i * N + j]) {
        if (DEBUG) {
          std::cout << "Error at " << i << " " << j << std::endl;
        }
        return false;
      }
    }
  }
  return true;
}

/////////////////// Main /////////////////

// Main
#define MPOW 10
#define NPOW 10
#define KPOW 10
int main() {
  std::cout << "Starting..." << std::endl;

  // Initialize two matrices
  float *A, *B, *C, *D;
  int M = 1 << MPOW;
  int N = 1 << NPOW;
  int K = 1 << KPOW;

  // Allocate the Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&A, M * K * sizeof(float));
  cudaMallocManaged(&B, K * N * sizeof(float));
  cudaMallocManaged(&C, M * N * sizeof(float));
  cudaMallocManaged(&D, M * N * sizeof(float));

  dim3 block_size(TPB, TPB);
  dim3 grid_size((M + block_size.x - 1) / block_size.x,
                 (N + block_size.y - 1) / block_size.y);

  // Initialize the matrices on the CPU
  init(A, M, K);
  init(B, K, N);

  // Run the matrix multiplication on the GPU
  matmul_kernel_v00<<<grid_size, block_size>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
  matmul_kernel_v01<<<grid_size, block_size>>>(A, B, D, M, N, K);
  cudaDeviceSynchronize();

  // Print the matrices if we are in debug mode
  if (DEBUG) {
    std::cout << "A:" << std::endl;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        std::cout << A[i * K + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "B:" << std::endl;
    for (int i = 0; i < K; i++) {
      for (int j = 0; j < N; j++) {
        std::cout << B[i * N + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "C:" << std::endl;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        std::cout << C[i * N + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "D:" << std::endl;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        std::cout << D[i * N + j] << " ";
      }
      std::cout << std::endl;
    }

    // Check that the matrixes are the same
    bool success = verify(C, D, M, N);
    if (!success) {
      std::cout << "Error in Matrix Multiplication!" << std::endl;
    } else {
      std::cout << "Success!" << std::endl;
    }
  }
  // Free the memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(D);

  // Print when done
  std::cout << "Done!" << std::endl;
  return 0;
}
