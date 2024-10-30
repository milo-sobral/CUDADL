// Contains some utility functions for the CUDADL library
// Author: Miolo

#include <iostream>
#include "../include/utils.h"

/////////////////// Multiplication function/////////////////

// We start by writing a simple matrix multiplication running on the CPU
// We have A: MxK, B: KxN, C: MxN
template <typename T> void matmul_stupid(T *A, T *B, T *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0;
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
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
