#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

// Constant definitions
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 1024;
const int ITERATIONS = 1000;

// Performance test configuration
struct TestConfig {
    int blockSize;
    int gridSize;
    int iterations;
    const char* name;
};

// Basic kernel - unoptimized version
__global__ void basicKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Simple computation operations
    if (idx < n) {
        float x = input[idx];
        output[idx] = x * x + x + 1.0f;
    }
}

// Optimized version 1: using register caching
__global__ void optimizedKernel1(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float temp[10];  // Use register array
        
        // Batch load data into registers
        for (int i = 0; i < 10; i++) {
            int load_idx = (idx + i * gridDim.x * blockDim.x) % n;
            temp[i] = input[load_idx];
        }
        
        // Perform computation in registers
        float result = 0.0f;
        for (int i = 0; i < 10; i++) {
            result += temp[i] * temp[i] + temp[i] + 1.0f;
        }
        
        // Merge results
        output[idx] = result / 10.0f;
    }
}

// Optimized version 2: using shared memory
__global__ void optimizedKernel2(float *input, float *output, int n) {
    __shared__ float s_data[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Cooperatively load data into shared memory
    if (idx < n) {
        s_data[tid] = input[idx];
    } else {
        s_data[tid] = 0.0f;
    }
    __syncthreads();
    
    // Perform computation in shared memory
    if (idx < n) {
        float x = s_data[tid];
        float result = x * x + x + 1.0f;
        
        // Cooperatively store results
        output[idx] = result;
    }
}

// Optimized version 3: using loop unrolling
__global__ void optimizedKernel3(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        float result = x;
        
        // Loop unrolling for better instruction-level parallelism
        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            result = result * result + result + 1.0f;
        }
        
        output[idx] = result;
    }
}

// Optimized version 4: using vectorized memory access
__global__ void optimizedKernel4(float4 *input, float4 *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float4 x = input[idx];
        float4 result;
        
        // Vectorized computation
        result.x = x.x * x.x + x.x + 1.0f;
        result.y = x.y * x.y + x.y + 1.0f;
        result.z = x.z * x.z + x.z + 1.0f;
        result.w = x.w * x.w + x.w + 1.0f;
        
        output[idx] = result;
    }
}

// Performance measurement function
void measurePerformance(const char* kernelName, void (*kernel)(float*, float*, int),
                       float *d_input, float *d_output, int n, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    // Measure performance
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("%s: %.3f ms per iteration, %.2f GFLOPS\n",
           kernelName, milliseconds / iterations,
           (float)(n * iterations * 16) / (milliseconds / 1000.0) / 1e9);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Vectorized performance measurement
void measureVectorizedPerformance(const char* kernelName, void (*kernel)(float4*, float4*, int),
                                float4 *d_input, float4 *d_output, int n, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    // Measure performance
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("%s: %.3f ms per iteration, %.2f GFLOPS\n",
           kernelName, milliseconds / iterations,
           (float)(n * 4 * iterations * 16) / (milliseconds / 1000.0) / 1e9);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== CUDA Performance Tuning Benchmark ===\n");
    
    // Get GPU information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("\n");
    
    // Test configurations
    TestConfig configs[] = {
        {256, 1024, ITERATIONS, "256 threads/block"},
        {512, 512, ITERATIONS, "512 threads/block"},
        {1024, 256, ITERATIONS, "1024 threads/block"},
        {128, 2048, ITERATIONS, "128 threads/block"}
    };
    
    for (const auto& config : configs) {
        printf("=== Testing %s ===\n", config.name);
        
        // Allocate memory
        size_t size = config.gridSize * config.blockSize * sizeof(float);
        float *h_input, *h_output;
        float *d_input, *d_output;
        
        cudaMallocHost(&h_input, size);
        cudaMallocHost(&h_output, size);
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);
        
        // Initialize data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (int i = 0; i < config.gridSize * config.blockSize; i++) {
            h_input[i] = dis(gen);
        }
        
        // Copy data to device
        cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
        
        // Test different kernels
        printf("Block Size: %d, Grid Size: %d\n", config.blockSize, config.gridSize);
        measurePerformance("Basic Kernel", basicKernel, d_input, d_output, 
                         config.gridSize * config.blockSize, config.iterations);
        measurePerformance("Optimized Kernel 1 (Register)", optimizedKernel1, d_input, d_output,
                         config.gridSize * config.blockSize, config.iterations);
        measurePerformance("Optimized Kernel 2 (Shared Memory)", optimizedKernel2, d_input, d_output,
                         config.gridSize * config.blockSize, config.iterations);
        measurePerformance("Optimized Kernel 3 (Loop Unrolling)", optimizedKernel3, d_input, d_output,
                         config.gridSize * config.blockSize, config.iterations);
        
        // Test vectorized version
        size_t vectorSize = config.gridSize * config.blockSize / 4;
        float4 *d_input4, *d_output4;
        cudaMalloc(&d_input4, vectorSize * sizeof(float4));
        cudaMalloc(&d_output4, vectorSize * sizeof(float4));
        
        // Convert float to float4
        cudaMemcpy(d_input4, h_input, vectorSize * sizeof(float4), cudaMemcpyHostToDevice);
        measureVectorizedPerformance("Optimized Kernel 4 (Vectorized)", optimizedKernel4, d_input4, d_output4,
                                   vectorSize, config.iterations);
        
        printf("\n");
        
        // Clean up
        cudaFreeHost(h_input);
        cudaFreeHost(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_input4);
        cudaFree(d_output4);
    }
    
    printf("=== Performance Tuning Tips ===\n");
    printf("1. Use appropriate thread block sizes (multiples of 32)\n");
    printf("2. Utilize shared memory for data reuse\n");
    printf("3. Use registers for frequently accessed data\n");
    printf("4. Consider loop unrolling for better ILP\n");
    printf("5. Use vectorized memory access when possible\n");
    printf("6. Profile with nvprof or Nsight Compute\n");
    printf("7. Monitor occupancy and resource usage\n");
    
    return 0;
}
