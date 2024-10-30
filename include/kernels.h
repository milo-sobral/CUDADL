#ifndef KERNELS_H
#define KERNELS_H

template <typename T>
void launch_matmul(T *A, T *B, T *C, int M, int N, int K, int version);

#endif

#ifndef TPB
#define TPB 32
#endif
