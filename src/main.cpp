#include "matrix_multiply.hpp"
#include "utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // Example matrix dimensions
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    // Allocate host memory
    float *h_A = new float[M * N];
    float *h_B = new float[N * K];
    float *h_C = new float[M * K];

    // Initialize matrices
    initializeMatrix(h_A, M, N);
    initializeMatrix(h_B, N, K);

    // Perform matrix multiplication
    matrixMultiply(h_A, h_B, h_C, M, N, K);

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}