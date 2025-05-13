#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

int main()
{
    const int M = 1000; // Number of rows in Matrix A and result matrix
    const int K = 1500; // Number of columns in Matrix A and rows in Matrix B
    const int N = 3000; // Number of columns in Matrix B and result matrix

    // Initialize matrices
    std::vector<std::vector<double>> A(M, std::vector<double>(K, 0.0));
    std::vector<std::vector<double>> B(K, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> C(M, std::vector<double>(N, 0.0));

    // Fill matrices with some values
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            A[i][j] = i + j;
        }
    }

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            B[i][j] = i * j;
        }
    }

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

// Matrix multiplication using OpenMP
#pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i][j] = 0.0;
            for (int k = 0; k < K; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // End measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Print execution time
    std::cout << "Matrix multiplication completed in " << elapsed.count()
              << " seconds" << std::endl;

    // Verify result (optional) - print a small portion of result matrix
    std::cout << "Sample of result matrix (first 3x3 elements):" << std::endl;
    for (int i = 0; i < std::min(3, M); i++)
    {
        for (int j = 0; j < std::min(3, N); j++)
        {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}