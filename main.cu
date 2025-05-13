#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Kernel function for matrix multiplication
__global__ void matrixMultKernel(const double *A, const double *B, double *C, int M, int K, int N)
{
    // Calculate global row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if within matrix bounds
    if (row < M && col < N)
    {
        double sum = 0.0;
        for (int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Function to check for CUDA errors
void checkCudaError(cudaError_t error, const char *msg)
{
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int M = 100; // Number of rows in Matrix A and result matrix
    const int K = 150; // Number of columns in Matrix A and rows in Matrix B
    const int N = 300; // Number of columns in Matrix B and result matrix

    // Allocate host memory
    double *h_A = new double[M * K];
    double *h_B = new double[K * N];
    double *h_C = new double[M * N];

    // Initialize matrices with values
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            h_A[i * K + j] = i + j;
        }
    }

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_B[i * N + j] = i * j;
        }
    }

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void **)&d_A, M * K * sizeof(double)), "cudaMalloc A");
    checkCudaError(cudaMalloc((void **)&d_B, K * N * sizeof(double)), "cudaMalloc B");
    checkCudaError(cudaMalloc((void **)&d_C, M * N * sizeof(double)), "cudaMalloc C");

    // Transfer data from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, M * K * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, K * N * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy B to device");

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch kernel
    matrixMultKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    // Wait for GPU to finish
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy C to host");

    // Print execution time
    std::cout << "CUDA matrix multiplication completed in " << elapsed.count()
              << " seconds" << std::endl;

    // Verify result (optional) - print a small portion of result matrix
    std::cout << "Sample of result matrix (first 3x3 elements):" << std::endl;
    for (int i = 0; i < std::min(3, M); i++)
    {
        for (int j = 0; j < std::min(3, N); j++)
        {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}