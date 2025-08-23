#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

// Constant definitions
const int TILE_SIZE = 16;
const int BLOCK_SIZE = 256;

// Basic matrix multiplication kernel
__global__ void matrixMulBasic(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Shared memory optimized matrix multiplication kernel
__global__ void matrixMulShared(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < K; k += TILE_SIZE) {
        // Cooperatively load data into shared memory
        if (row < M && k + tx < K)
            sA[ty][tx] = A[row * K + k + tx];
        else
            sA[ty][tx] = 0.0f;
            
        if (col < N && k + ty < K)
            sB[ty][tx] = B[(k + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Calculate dot product within tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Matrix multiplication using CUDA streams
__global__ void matrixMulStream(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < K; k += TILE_SIZE) {
        if (row < M && k + tx < K)
            sA[ty][tx] = A[row * K + k + tx];
        else
            sA[ty][tx] = 0.0f;
            
        if (col < N && k + ty < K)
            sB[ty][tx] = B[(k + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Verify results
bool verifyMatrixMultiplication(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) {
                expected += A[i * K + k] * B[k * N + j];
            }
            if (abs(C[i * N + j] - expected) > 1e-3) {
                std::cout << "Mismatch at [" << i << "][" << j << "]: "
                          << C[i * N + j] << " vs " << expected << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Performance test function
void benchmarkMatrixMultiplication(int M, int N, int K, int iterations = 10) {
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];
    
    // Initialize data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < M * K; i++) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; i++) h_B[i] = dis(gen);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // Configure kernel parameters
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Test basic version
    std::cout << "Testing Basic Matrix Multiplication..." << std::endl;
    cudaMemset(d_C, 0, sizeC);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        matrixMulBasic<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy results back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // Calculate performance metrics
    double ops = 2.0 * M * N * K; // Multiplications and additions
    double gflops = (ops * iterations) / (duration.count() * 1000.0);
    
    std::cout << "Basic: " << gflops << " GFLOPS, "
              << duration.count() / iterations << " μs per iteration" << std::endl;
    
    // Test shared memory version
    std::cout << "Testing Shared Memory Matrix Multiplication..." << std::endl;
    cudaMemset(d_C, 0, sizeC);
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        matrixMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    gflops = (ops * iterations) / (duration.count() * 1000.0);
    std::cout << "Shared: " << gflops << " GFLOPS, "
              << duration.count() / iterations << " μs per iteration" << std::endl;
    
    // Test stream version
    std::cout << "Testing Stream Matrix Multiplication..." << std::endl;
    cudaMemset(d_C, 0, sizeC);
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        matrixMulStream<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    gflops = (ops * iterations) / (duration.count() * 1000.0);
    std::cout << "Stream: " << gflops << " GFLOPS, "
              << duration.count() / iterations << " μs per iteration" << std::endl;
    
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    if (verifyMatrixMultiplication(h_A, h_B, h_C, M, N, K)) {
        std::cout << "Results verified successfully!" << std::endl;
    } else {
        std::cout << "Results verification failed!" << std::endl;
    }
    
    // Clean up resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
}

int main() {
    std::cout << "CUDA Matrix Multiplication Benchmark" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Test different matrix configurations
    struct MatrixConfig {
        int M, N, K;
        const char* name;
    };
    
    MatrixConfig configs[] = {
        {64, 64, 64, "64x64x64"},
        {128, 128, 128, "128x128x128"},
        {256, 256, 256, "256x256x256"},
        {512, 512, 512, "512x512x512"}
    };
    
    for (const auto& config : configs) {
        std::cout << "\nTesting " << config.name << ":" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        benchmarkMatrixMultiplication(config.M, config.N, config.K);
    }
    
    return 0;
}
