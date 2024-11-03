#ifndef MATRIX_MULTIPLY_HPP
#define MATRIX_MULTIPLY_HPP

#include <cuda_runtime.h>
#include <stdio.h>

/**
 * @brief Block size for CUDA kernel execution
 * This defines the dimensions of thread blocks (16x16 threads per block)
 */
#define BLOCK_SIZE 16

/**
 * @brief CUDA kernel for matrix multiplication C = A * B
 * 
 * @param A Input matrix A of size M x N in device memory
 * @param B Input matrix B of size N x K in device memory
 * @param C Output matrix C of size M x K in device memory
 * @param M Number of rows in matrix A and C
 * @param N Number of columns in A and rows in B
 * @param K Number of columns in matrix B and C
 * 
 * @note This kernel must be launched with appropriate grid and block dimensions
 */
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int M, int N, int K);

/**
 * @brief Host function to perform matrix multiplication C = A * B using CUDA
 * 
 * @param h_A Input matrix A in host memory (size M x N)
 * @param h_B Input matrix B in host memory (size N x K)
 * @param h_C Output matrix C in host memory (size M x K)
 * @param M Number of rows in matrix A and C
 * @param N Number of columns in A and rows in B
 * @param K Number of columns in matrix B and C
 * 
 * @note This function handles:
 *       - Memory allocation on GPU
 *       - Data transfer between CPU and GPU
 *       - Kernel execution
 *       - Error handling
 *       - Memory cleanup
 * 
 * @requirements
 *       - CUDA-capable GPU
 *       - Sufficient GPU memory for matrices
 *       - Valid input matrices with compatible dimensions
 */
void matrixMultiply(float* h_A, float* h_B, float* h_C, int M, int N, int K);

#endif // MATRIX_MULTIPLY_HPP