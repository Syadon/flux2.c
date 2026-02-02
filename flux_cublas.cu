/*
 * FLUX cuBLAS Wrapper - Implementation
 *
 * Provides cuBLAS GPU acceleration for NVIDIA GPUs.
 * Manages CUDA memory transfers and cuBLAS operations.
 */

#include "flux_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

/* Global cuBLAS handle */
static cublasHandle_t cublas_handle = NULL;
static int cublas_initialized = 0;

/* Workspace for GPU memory - reused to avoid allocation overhead */
static float *d_A = NULL;
static float *d_B = NULL;
static float *d_C = NULL;
static size_t d_A_size = 0;
static size_t d_B_size = 0;
static size_t d_C_size = 0;

/* Helper to check CUDA errors */
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_RET(call, ret) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return ret; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        return; \
    } \
} while(0)

int flux_cublas_init(void) {
    if (cublas_initialized) {
        return 1;
    }

    /* Check for CUDA device */
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "cuBLAS: No CUDA devices found\n");
        return 0;
    }

    /* Get device info */
    cudaDeviceProp prop;
    CUDA_CHECK_RET(cudaGetDeviceProperties(&prop, 0), 0);
    fprintf(stderr, "cuBLAS: Using %s (%.1f GB, compute %d.%d)\n",
            prop.name,
            (float)prop.totalGlobalMem / (1024 * 1024 * 1024),
            prop.major, prop.minor);

    /* Create cuBLAS handle */
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS: Failed to create handle (error %d)\n", status);
        return 0;
    }

    /* Set math mode for best performance */
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    cublas_initialized = 1;
    fprintf(stderr, "cuBLAS: GPU acceleration enabled\n");
    return 1;
}

void flux_cublas_cleanup(void) {
    if (d_A) { cudaFree(d_A); d_A = NULL; d_A_size = 0; }
    if (d_B) { cudaFree(d_B); d_B = NULL; d_B_size = 0; }
    if (d_C) { cudaFree(d_C); d_C = NULL; d_C_size = 0; }

    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = NULL;
    }
    cublas_initialized = 0;
}

int flux_cublas_available(void) {
    return cublas_initialized;
}

/* Ensure GPU buffer is large enough */
static int ensure_buffer(float **d_buf, size_t *current_size, size_t needed_size) {
    if (*current_size >= needed_size) {
        return 1;
    }

    if (*d_buf) {
        cudaFree(*d_buf);
    }

    /* Allocate with some extra room to avoid frequent reallocations */
    size_t alloc_size = needed_size + needed_size / 4;
    cudaError_t err = cudaMalloc((void**)d_buf, alloc_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuBLAS: Failed to allocate %.1f MB GPU memory\n",
                (float)alloc_size / (1024 * 1024));
        *d_buf = NULL;
        *current_size = 0;
        return 0;
    }
    *current_size = alloc_size;
    return 1;
}

void flux_cublas_sgemm(int transpose_a, int transpose_b,
                       int M, int N, int K,
                       float alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       float beta,
                       float *C, int ldc) {
    if (!cublas_initialized) {
        fprintf(stderr, "cuBLAS: Not initialized\n");
        return;
    }

    /* Calculate sizes */
    size_t size_A = (transpose_a ? K : M) * lda * sizeof(float);
    size_t size_B = (transpose_b ? N : K) * ldb * sizeof(float);
    size_t size_C = M * ldc * sizeof(float);

    /* Ensure GPU buffers are large enough */
    if (!ensure_buffer(&d_A, &d_A_size, size_A) ||
        !ensure_buffer(&d_B, &d_B_size, size_B) ||
        !ensure_buffer(&d_C, &d_C_size, size_C)) {
        return;
    }

    /* Copy input data to GPU */
    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
    if (beta != 0.0f) {
        CUDA_CHECK(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice));
    }

    /*
     * cuBLAS uses column-major order, but our data is row-major.
     * We use the identity: C = A @ B  <==>  C^T = B^T @ A^T
     * So we swap A and B and their transpose flags.
     */
    cublasOperation_t op_a = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    /* Note: dimensions are swapped for column-major */
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                             op_a, op_b,
                             N, M, K,
                             &alpha,
                             d_B, ldb,
                             d_A, lda,
                             &beta,
                             d_C, ldc));

    /* Copy result back to CPU */
    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));
}
