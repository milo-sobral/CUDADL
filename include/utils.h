#ifndef UTILS_H
#define UTILS_H

template <typename T> void matmul_stupid(T *A, T *B, T *C, int M, int N, int K); 
template <typename T> void init(T *A, int M, int N);
template <typename T> bool verify(T *A, T *B, int M, int N); 

#endif

#ifndef DEBUG
#define DEBUG false
#endif
