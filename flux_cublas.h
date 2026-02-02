/*
 * FLUX cuBLAS Wrapper - Header
 *
 * Provides cuBLAS GPU acceleration for NVIDIA GPUs.
 * This wrapper manages CUDA memory and cuBLAS handles transparently.
 */

#ifndef FLUX_CUBLAS_H
#define FLUX_CUBLAS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize cuBLAS - must be called before any cuBLAS operations */
int flux_cublas_init(void);

/* Cleanup cuBLAS resources */
void flux_cublas_cleanup(void);

/* Check if cuBLAS is available and initialized */
int flux_cublas_available(void);

/*
 * cuBLAS SGEMM wrapper - matches cblas_sgemm semantics but uses GPU
 * C = alpha * op(A) * op(B) + beta * C
 * 
 * transpose_a: 0=no transpose, 1=transpose
 * transpose_b: 0=no transpose, 1=transpose
 * M, N, K: matrix dimensions
 * alpha, beta: scalar multipliers
 * A: [M, K] or [K, M] if transposed
 * B: [K, N] or [N, K] if transposed
 * C: [M, N]
 * lda, ldb, ldc: leading dimensions
 */
void flux_cublas_sgemm(int transpose_a, int transpose_b,
                       int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc);

#ifdef __cplusplus
}
#endif

#endif /* FLUX_CUBLAS_H */
