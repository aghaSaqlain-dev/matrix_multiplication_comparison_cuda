#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>

#define N 2048  // Adjust according to your system's memory

int main() {
    // Allocate memory for matrices
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
        C[i] = 0.0f;
    }

    // Measure execution time
    double start = omp_get_wtime();

    // Parallel matrix multiplication using OpenMP
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }

    double end = omp_get_wtime();
    std::cout << "Execution time with OpenMP: " << (end - start) * 1000 << " ms\n";


    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
