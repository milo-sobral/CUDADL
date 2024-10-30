// This file is used to test the CUDADL library
// Author: Miolo

#include <iostream>

#include "../include/main.h"

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
