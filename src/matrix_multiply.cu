
#include<iostream>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "config.hpp"  // For BLOCK_SIZE definition
#include "utils.hpp"   // For utility functions


using namespace std;
// CUDA kernel for matrix multiplication
// Multiplies matrix A (M x N) with matrix B (N x K) to produce matrix C (M x K)
// this is for each cell in the matrix C
// a thread is responsible for computing the value of a single cell in the matrix C
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int M, int N, int K) {
    // Compute the row and column indices of the element C(row, col) to be computed
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within matrix bounds
    if (row < M && col < K) {
        float sum = 0.0f;  // Accumulator for dot product
        // Perform dot product for the row of A and column of B
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;  // Store the computed value in matrix C
    }
}


// Host function to perform matrix multiplication using CUDA
void matrixMultiply(float* h_A, float* h_B, float* h_C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;  // Device memory pointers

    // Calculate sizes of matrices A, B, and C
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    // Error variable to track CUDA operations
    cudaError_t error = cudaSuccess;

    // Allocate device memory for matrix A and check for errors
    error = cudaMalloc(&d_A, size_A);
    if (error != cudaSuccess) {
        printf("Error allocating device memory for A: %s\n", cudaGetErrorString(error));
        return;
    }

    // Allocate device memory for matrix B and check for errors
    error = cudaMalloc(&d_B, size_B);
    if (error != cudaSuccess) {
        cudaFree(d_A);  // Free previously allocated memory
        printf("Error allocating device memory for B: %s\n", cudaGetErrorString(error));
        return;
    }

    // Allocate device memory for matrix C and check for errors
    error = cudaMalloc(&d_C, size_C);
    if (error != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        printf("Error allocating device memory for C: %s\n", cudaGetErrorString(error));
        return;
    }

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Configure CUDA kernel launch dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                 (M + dimBlock.y - 1) / dimBlock.y);

    // Launch CUDA kernel to perform matrix multiplication
    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // Check for any errors in kernel launch
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(error));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        cerr << "Error: Unable to get device count: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    cout << "Number of available GPUs: " << deviceCount << std::endl;

    int gpu_id = 0;  // Select GPU 1 (second GPU)
    if (gpu_id < deviceCount){
        // Set the desired GPU
        cudaSetDevice(gpu_id);
        cout << "Using GPU: " << gpu_id << std::endl;
    }

    const int M = 64;  // Number of rows in matrix A and matrix C
    const int N = 32;  // Number of columns in matrix A and rows in matrix B
    const int K = 128; // Number of columns in matrix B and matrix C

    // Allocate host memory for matrices A, B, and C
    float* h_A = (float*)malloc(M * N * sizeof(float));
    float* h_B = (float*)malloc(N * K * sizeof(float));
    float* h_C = (float*)malloc(M * K * sizeof(float));

    // Initialize matrices A and B with random values
    initializeMatrix(h_A, M, N);
    initializeMatrix(h_B, N, K);

    // Display initial values for matrices A and B (first 2x2 section)
    printf("Matrix A (first 2x2):\n");
    printMatrix(h_A, 2, 2);
    printf("Matrix B (first 2x2):\n");
    printMatrix(h_B, 2, 2);

    // Perform matrix multiplication using CUDA
    matrixMultiply(h_A, h_B, h_C, M, N, K);

    // Display the result of matrix multiplication (first 2x2 section of C)
    printf("Result Matrix C (first 2x2):\n");
    printMatrix(h_C, 2, 2);

    // Free host memory for matrices A, B, and C
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}