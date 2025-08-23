#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cstring>

// 常量定义
const int TILE_SIZE = 16;
const int MAX_SEQ_LEN = 1024;
const int MAX_D_K = 512;

// 性能测试配置
struct PerformanceConfig {
    int dataSize;
    int iterations;
    const char* name;
};

// 基础kernel - 未优化版本
__global__ void basicKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 简单的计算操作
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = sinf(val) + cosf(val);
        }
        data[idx] = val;
    }
}

// 优化版本1：使用寄存器缓存
__global__ void registerOptimizedKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        float temp[10];  // 使用寄存器数组
        
        // 批量加载数据到寄存器
        for (int i = 0; i < 10; i++) {
            temp[i] = val + i;
        }
        
        // 在寄存器中进行计算
        for (int i = 0; i < 10; i++) {
            temp[i] = sinf(temp[i]) + cosf(temp[i]);
        }
        
        // 合并结果
        float sum = 0.0f;
        for (int i = 0; i < 10; i++) {
            sum += temp[i];
        }
        
        data[idx] = sum;
    }
}

// 优化版本2：使用共享内存
__global__ void sharedMemoryKernel(float *data, int n) {
    __shared__ float shared_cache[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        // 协作加载数据到共享内存
        shared_cache[tid] = data[idx];
        __syncthreads();
        
        // 在共享内存中进行计算
        float val = shared_cache[tid];
        for (int i = 0; i < 100; i++) {
            val = sinf(val) + cosf(val);
        }
        
        // 协作存储结果
        shared_cache[tid] = val;
        __syncthreads();
        
        data[idx] = shared_cache[tid];
    }
}

// 优化版本3：循环展开
__global__ void loopUnrolledKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        
        // 手动循环展开
        val = sinf(val) + cosf(val);
        val = sinf(val) + cosf(val);
        val = sinf(val) + cosf(val);
        val = sinf(val) + cosf(val);
        val = sinf(val) + cosf(val);
        
        val = sinf(val) + cosf(val);
        val = sinf(val) + cosf(val);
        val = sinf(val) + cosf(val);
        val = sinf(val) + cosf(val);
        val = sinf(val) + cosf(val);
        
        data[idx] = val;
    }
}

// 优化版本4：减少分支分歧
__global__ void branchOptimizedKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        
        // 使用条件运算符减少分支分歧
        float result = (val > 0) ? sqrtf(val) : 0.0f;
        result = (val < 1.0f) ? result * 2.0f : result;
        result = (val > 2.0f) ? result / 2.0f : result;
        
        data[idx] = result;
    }
}

// 优化版本5：向量化操作
__global__ void vectorizedKernel(float4 *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float4 val = data[idx];
        
        // 向量化操作
        val.x = sinf(val.x) + cosf(val.x);
        val.y = sinf(val.y) + cosf(val.y);
        val.z = sinf(val.z) + cosf(val.z);
        val.w = sinf(val.w) + cosf(val.w);
        
        data[idx] = val;
    }
}

// 矩阵转置优化示例
__global__ void matrixTransposeNaive(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[y * width + x] = input[x * height + y];
    }
}

__global__ void matrixTransposeShared(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // 协作加载数据到共享内存
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // 协作写入输出
    int newX = blockIdx.y * TILE_SIZE + threadIdx.x;
    int newY = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (newX < height && newY < width) {
        output[newY * height + newX] = tile[threadIdx.x][threadIdx.y];
    }
}

// 内存合并访问优化示例
__global__ void memoryCoalescingBad(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = 256;  // 大步长，导致内存访问不合并
    
    if (idx < n) {
        data[idx * stride] = idx;  // 内存访问间隔很大
    }
}

__global__ void memoryCoalescingGood(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = idx;  // 连续的内存访问
    }
}

// 共享内存Bank冲突优化
__global__ void bankConflictKernel(float *input, float *output, int n) {
    __shared__ float shared_data[1024];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 使用交错索引避免Bank冲突
        int shared_idx = (tid * 33) % 1024;  // 33是质数，避免Bank冲突
        shared_data[shared_idx] = input[idx];
        __syncthreads();
        
        output[idx] = shared_data[shared_idx];
    }
}

// 性能测试函数
void benchmarkKernel(const char* kernelName, 
                     void (*kernel)(float*, int), 
                     float *d_data, int n, 
                     int iterations) {
    // 预热
    for (int i = 0; i < 5; i++) {
        kernel<<<(n + 255) / 256, 256>>>(d_data, n);
    }
    cudaDeviceSynchronize();
    
    // 性能测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<(n + 255) / 256, 256>>>(d_data, n);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 计算性能指标
    double ops = n * iterations * 200;  // 估算操作数
    double gops = ops / (milliseconds * 1000000.0);
    
    printf("%-25s: %8.2f ms, %8.2f GOPS\n", 
           kernelName, milliseconds / iterations, gops);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 矩阵转置性能测试
void benchmarkMatrixTranspose(int width, int height, int iterations) {
    size_t size = width * height * sizeof(float);
    
    // 分配内存
    float *h_input = new float[width * height];
    float *h_output = new float[width * height];
    float *d_input, *d_output;
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // 初始化数据
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)i;
    }
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // 配置kernel参数
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE, 
                  (height + TILE_SIZE - 1) / TILE_SIZE);
    
    // 测试基础版本
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        matrixTransposeNaive<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Matrix Transpose Naive: %8.2f ms\n", milliseconds / iterations);
    
    // 测试共享内存版本
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        matrixTransposeShared<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Matrix Transpose Shared: %8.2f ms\n", milliseconds / iterations);
    
    // 清理资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
}

// 获取最优线程块大小
int getOptimalBlockSize() {
    cudaDeviceProp prop;
    cudaGetDevice(&prop);
    
    // 考虑共享内存限制
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int maxSharedMemoryPerSM = prop.sharedMemoryPerMultiprocessor;
    
    // 考虑寄存器限制
    int maxRegistersPerSM = prop.regsPerMultiprocessor;
    
    // 返回最优配置
    return min(256, maxThreadsPerSM / prop.multiProcessorCount);
}

// 获取最优网格大小
dim3 getOptimalGridSize(int n, int blockSize) {
    int blocksX = (n + blockSize - 1) / blockSize;
    int blocksY = 1;
    int blocksZ = 1;
    
    // 考虑GPU的SM数量
    cudaDeviceProp prop;
    cudaGetDevice(&prop);
    int maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
    int totalBlocks = blocksX * blocksY * blocksZ;
    
    // 确保有足够的SM来并行执行
    if (totalBlocks < prop.multiProcessorCount * maxBlocksPerSM) {
        blocksX = max(blocksX, prop.multiProcessorCount);
    }
    
    return dim3(blocksX, blocksY, blocksZ);
}

int main() {
    std::cout << "CUDA Performance Tuning Benchmark" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // 显示GPU信息
    cudaDeviceProp prop;
    cudaGetDevice(&prop);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Shared Memory per SM: %zu KB\n", prop.sharedMemoryPerMultiprocessor / 1024);
    printf("Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("\n");
    
    // 性能测试配置
    PerformanceConfig configs[] = {
        {1000000, 100, "1M elements"},
        {10000000, 50, "10M elements"},
        {100000000, 20, "100M elements"}
    };
    
    for (const auto& config : configs) {
        printf("Testing %s:\n", config.name);
        printf("----------------------------------------\n");
        
        // 分配内存
        size_t size = config.dataSize * sizeof(float);
        float *h_data = new float[config.dataSize];
        float *d_data;
        
        cudaMalloc(&d_data, size);
        
        // 初始化数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 10.0f);
        
        for (int i = 0; i < config.dataSize; i++) {
            h_data[i] = dis(gen);
        }
        
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
        
        // 测试不同版本的kernel
        benchmarkKernel("Basic", basicKernel, d_data, config.dataSize, config.iterations);
        benchmarkKernel("Register Optimized", registerOptimizedKernel, d_data, config.dataSize, config.iterations);
        benchmarkKernel("Shared Memory", sharedMemoryKernel, d_data, config.dataSize, config.iterations);
        benchmarkKernel("Loop Unrolled", loopUnrolledKernel, d_data, config.dataSize, config.iterations);
        benchmarkKernel("Branch Optimized", branchOptimizedKernel, d_data, config.dataSize, config.iterations);
        
        // 测试向量化版本（如果数据大小是4的倍数）
        if (config.dataSize % 4 == 0) {
            float4 *d_data4 = (float4*)d_data;
            benchmarkKernel("Vectorized (float4)", 
                           (void(*)(float*, int))vectorizedKernel, 
                           (float*)d_data4, config.dataSize / 4, config.iterations);
        }
        
        printf("\n");
        
        // 清理资源
        cudaFree(d_data);
        delete[] h_data;
    }
    
    // 测试矩阵转置
    printf("Matrix Transpose Performance Test:\n");
    printf("----------------------------------------\n");
    benchmarkMatrixTranspose(1024, 1024, 100);
    printf("\n");
    
    // 测试内存合并访问
    printf("Memory Coalescing Test:\n");
    printf("----------------------------------------\n");
    
    int n = 1000000;
    size_t size = n * sizeof(float);
    float *h_data = new float[n];
    float *d_data;
    
    cudaMalloc(&d_data, size);
    
    // 测试基础版本
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        memoryCoalescingBad<<<(n + 255) / 256, 256>>>(d_data, n);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Memory Coalescing Bad: %8.2f ms\n", milliseconds / 100);
    
    // 测试优化版本
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        memoryCoalescingGood<<<(n + 255) / 256, 256>>>(d_data, n);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Memory Coalescing Good: %8.2f ms\n", milliseconds / 100);
    
    // 测试Bank冲突优化
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        bankConflictKernel<<<(n + 255) / 256, 256>>>(d_data, d_data, n);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Bank Conflict Optimized: %8.2f ms\n", milliseconds / 100);
    
    // 显示最优配置
    printf("\nOptimal Configuration:\n");
    printf("----------------------------------------\n");
    printf("Optimal Block Size: %d\n", getOptimalBlockSize());
    
    dim3 optimalGrid = getOptimalGridSize(1000000, 256);
    printf("Optimal Grid Size: (%d, %d, %d)\n", 
           optimalGrid.x, optimalGrid.y, optimalGrid.z);
    
    // 清理资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    delete[] h_data;
    
    printf("\nPerformance tuning benchmark completed!\n");
    
    return 0;
}
