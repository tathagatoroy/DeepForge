#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>  // For blockIdx, threadIdx, etc.

#define BLOCK_SIZE 32

// Add __global__ qualifier for CUDA kernel
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

void matrixMultiply(float* h_A, float* h_B, float* h_C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);
    
    // Error checking for CUDA operations
    cudaError_t error = cudaSuccess;
    
    error = cudaMalloc(&d_A, size_A);
    if (error != cudaSuccess) {
        printf("Error allocating device memory for A: %s\n", cudaGetErrorString(error));
        return;
    }
    
    error = cudaMalloc(&d_B, size_B);
    if (error != cudaSuccess) {
        cudaFree(d_A);
        printf("Error allocating device memory for B: %s\n", cudaGetErrorString(error));
        return;
    }
    
    error = cudaMalloc(&d_C, size_C);
    if (error != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        printf("Error allocating device memory for C: %s\n", cudaGetErrorString(error));
        return;
    }
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                 (M + dimBlock.y - 1) / dimBlock.y);
    
    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(error));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    const int M = 64;
    const int N = 32;
    const int K = 128;
    
    float* h_A = (float*)malloc(M * N * sizeof(float));
    float* h_B = (float*)malloc(N * K * sizeof(float));
    float* h_C = (float*)malloc(M * K * sizeof(float));
    
    initializeMatrix(h_A, M, N);
    initializeMatrix(h_B, N, K);
    
    printf("Matrix A (first 2x2):\n");
    printMatrix(h_A, 2, 2);
    printf("Matrix B (first 2x2):\n");
    printMatrix(h_B, 2, 2);
    
    matrixMultiply(h_A, h_B, h_C, M, N, K);
    
    printf("Result Matrix C (first 2x2):\n");
    printMatrix(h_C, 2, 2);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}