#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

// 常量定义
const int TILE_SIZE = 16;
const int BLOCK_SIZE = 256;

// 基础矩阵乘法kernel
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

// 共享内存优化的矩阵乘法kernel
__global__ void matrixMulShared(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < K; k += TILE_SIZE) {
        // 协作加载数据到共享内存
        if (row < M && k + tx < K)
            sA[ty][tx] = A[row * K + k + tx];
        else
            sA[ty][tx] = 0.0f;
            
        if (col < N && k + ty < K)
            sB[ty][tx] = B[(k + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // 计算tile内的点积
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 使用CUDA流的矩阵乘法
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

// 验证结果
bool verifyResult(float *A, float *B, float *C, int M, int N, int K) {
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

// 性能测试函数
void benchmarkMatrixMul(int M, int N, int K, int iterations = 10) {
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // 分配主机内存
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];
    
    // 初始化数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < M * K; i++) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; i++) h_B[i] = dis(gen);
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    // 复制数据到设备
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // 配置kernel参数
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // 测试基础版本
    std::cout << "Testing Basic Matrix Multiplication..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        matrixMulBasic<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 复制结果回主机
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // 计算性能指标
    double ops = 2.0 * M * N * K; // 乘法和加法
    double gflops = (ops * iterations) / (duration.count() * 1000.0);
    
    std::cout << "Basic: " << gflops << " GFLOPS, "
              << duration.count() / iterations << " μs per iteration" << std::endl;
    
    // 测试共享内存版本
    std::cout << "Testing Shared Memory Matrix Multiplication..." << std::endl;
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
    
    // 测试CUDA流版本
    std::cout << "Testing CUDA Stream Matrix Multiplication..." << std::endl;
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        // 在流1中处理
        cudaMemcpyAsync(d_A, h_A, sizeA, cudaMemcpyHostToDevice, stream1);
        matrixMulStream<<<gridDim, blockDim, 0, stream1>>>(d_A, d_B, d_C, M, N, K);
        cudaMemcpyAsync(h_C, d_C, sizeC, cudaMemcpyDeviceToHost, stream1);
        
        // 在流2中处理（如果有足够的数据）
        if (i < iterations - 1) {
            cudaMemcpyAsync(d_A, h_A, sizeA, cudaMemcpyHostToDevice, stream2);
            matrixMulStream<<<gridDim, blockDim, 0, stream2>>>(d_A, d_B, d_C, M, N, K);
            cudaMemcpyAsync(h_C, d_C, sizeC, cudaMemcpyDeviceToHost, stream2);
        }
    }
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    gflops = (ops * iterations) / (duration.count() * 1000.0);
    std::cout << "Stream: " << gflops << " GFLOPS, "
              << duration.count() / iterations << " μs per iteration" << std::endl;
    
    // 验证结果
    std::cout << "Verifying results..." << std::endl;
    if (verifyResult(h_A, h_B, h_C, M, N, K)) {
        std::cout << "Results verified successfully!" << std::endl;
    } else {
        std::cout << "Results verification failed!" << std::endl;
    }
    
    // 清理资源
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
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
    std::cout << "===================================" << std::endl;
    
    // 测试不同大小的矩阵
    int sizes[] = {512, 1024, 2048};
    
    for (int size : sizes) {
        std::cout << "\nTesting " << size << "x" << size << " matrices:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        benchmarkMatrixMul(size, size, size);
    }
    
    return 0;
}
