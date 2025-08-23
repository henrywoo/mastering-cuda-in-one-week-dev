/*
 * Mixed Precision Attention Mechanism Implementation
 * 
 * Compilation options (recommended to use compute-sanitizer for debugging memory issues):
 * 
 * 1. Normal compilation:
 *    nvcc -O3 -o mixed_precision_attention mixed_precision_attention.cu
 * 
 * 2. Using compute-sanitizer to debug memory issues:
 *    nvcc -g -G -o mixed_precision_attention mixed_precision_attention.cu
 *    compute-sanitizer --tool memcheck ./mixed_precision_attention
 * 
 * 3. Using compute-sanitizer to check uninitialized memory:
 *    compute-sanitizer --tool initcheck ./mixed_precision_attention
 * 
 * 4. Using compute-sanitizer to check race conditions:
 *    compute-sanitizer --tool racecheck ./mixed_precision_attention
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>
#include <random>

// Constant definitions
#define MAX_SEQ_LEN 2048  // Maximum sequence length

// Complete standard precision attention Kernel
__global__ void standardPrecisionAttentionKernel(float *Q, float *K, float *V, 
                                                float *output, float scale,
                                                int batchSize, int seqLen, int d_k, int d_v, int numHeads) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    
    // Boundary check
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= numHeads) {
        return;
    }
    
    // Calculate starting indices for current sequence position in Q, K, V
    int q_start = batchIdx * seqLen * d_k + seqIdx * d_k;
    int out_start = batchIdx * seqLen * d_v + seqIdx * d_v;
    
    // Calculate attention scores: dot product of Q with K at all sequence positions
    float max_score = -INFINITY;
    float attention_scores[MAX_SEQ_LEN];
    
    // Step 1: Calculate Q-K dot product and find maximum (for numerical stability)
    for (int pos = 0; pos < seqLen; pos++) {
        float score = 0.0f;
        int k_pos_start = batchIdx * seqLen * d_k + pos * d_k;
        
        // Calculate dot product of Q[seqIdx] with K[pos]
        for (int dim = 0; dim < d_k; dim++) {
            int q_idx = q_start + dim;
            int k_idx = k_pos_start + dim;
            
            if (q_idx < batchSize * seqLen * d_k && k_idx < batchSize * seqLen * d_k) {
                score += Q[q_idx] * K[k_idx];
            }
        }
        
        score *= scale;  // Apply scaling factor
        attention_scores[pos] = score;
        max_score = max(max_score, score);
    }
    
    // Step 2: Calculate softmax (numerically stable version)
    float sum_exp = 0.0f;
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] = expf(attention_scores[pos] - max_score);
        sum_exp += attention_scores[pos];
    }
    
    // Normalization
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] /= sum_exp;
    }
    
    // Step 3: Calculate weighted output
    for (int dim = 0; dim < d_v; dim++) {
        float weighted_sum = 0.0f;
        
        for (int pos = 0; pos < seqLen; pos++) {
            int v_idx = batchIdx * seqLen * d_v + pos * d_v + dim;
            if (v_idx < batchSize * seqLen * d_v) {
                weighted_sum += attention_scores[pos] * V[v_idx];
            }
        }
        
        int out_idx = out_start + dim;
        if (out_idx < batchSize * seqLen * d_v) {
            output[out_idx] = weighted_sum;
        }
    }
}

// Complete mixed precision attention Kernel
__global__ void mixedPrecisionAttentionKernel(half *Q, half *K, half *V, 
                                             float *output, float scale,
                                             int batchSize, int seqLen, int d_k, int d_v, int numHeads) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    
    // Boundary check
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= numHeads) {
        return;
    }
    
    // Calculate starting indices for current sequence position in Q, K, V
    int q_start = batchIdx * seqLen * d_k + seqIdx * d_k;
    int out_start = batchIdx * seqLen * d_v + seqIdx * d_v;
    
    // Calculate attention scores: dot product of Q with K at all sequence positions
    float max_score = -INFINITY;
    float attention_scores[MAX_SEQ_LEN];
    
    // Step 1: Calculate Q-K dot product and find maximum (for numerical stability)
    for (int pos = 0; pos < seqLen; pos++) {
        float score = 0.0f;
        int k_pos_start = batchIdx * seqLen * d_k + pos * d_k;
        
        // Calculate dot product of Q[seqIdx] with K[pos] (using FP16)
        for (int dim = 0; dim < d_k; dim++) {
            int q_idx = q_start + dim;
            int k_idx = k_pos_start + dim;
            
            if (q_idx < batchSize * seqLen * d_k && k_idx < batchSize * seqLen * d_k) {
                score += __half2float(Q[q_idx]) * __half2float(K[k_idx]);
            }
        }
        
        score *= scale;  // Apply scaling factor
        attention_scores[pos] = score;
        max_score = max(max_score, score);
    }
    
    // Step 2: Calculate softmax (numerically stable version)
    float sum_exp = 0.0f;
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] = expf(attention_scores[pos] - max_score);
        sum_exp += attention_scores[pos];
    }
    
    // Normalization
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] /= sum_exp;
    }
    
    // Step 3: Calculate weighted output
    for (int dim = 0; dim < d_v; dim++) {
        float weighted_sum = 0.0f;
        
        for (int pos = 0; pos < seqLen; pos++) {
            int v_idx = batchIdx * seqLen * d_v + pos * d_v + dim;
            if (v_idx < batchSize * seqLen * d_v) {
                weighted_sum += attention_scores[pos] * __half2float(V[v_idx]);
            }
        }
        
        int out_idx = out_start + dim;
        if (out_idx < batchSize * seqLen * d_v) {
            output[out_idx] = weighted_sum;
        }
    }
}

// Data conversion kernel: float -> half
__global__ void convertFloatToHalfKernel(float *input, half *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    bool debug_mode = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-d") == 0) {
            debug_mode = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --debug, -d    Enable debug mode, show detailed information\n");
            printf("  --help, -h     Show this help information\n");
            return 0;
        }
    }
    
    if (debug_mode) {
        printf("=== Debug mode enabled ===\n");
    }
    
    // Initialize CUDA
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        printf("CUDA device initialization failed: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("=== Mixed Precision Attention Test ===\n");
    
    // Get GPU information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    
    // Check if half precision is supported
    if (prop.major < 6) {
        printf("Warning: Current GPU does not support half precision, will use float precision simulation\n");
    }
    printf("\n");
    
    // Test parameters - use different d_k and d_v values to test indexing logic
    int batchSize = 1;
    int seqLen = 64;
    int d_k = 32;    // Different from d_v to test index calculation
    int d_v = 64;    // Different from d_k to test index calculation
    int numHeads = 4;
    int iterations = 100;
    
    printf("Test configuration: batch=%d, seq_len=%d, d_k=%d, d_v=%d, heads=%d\n\n", 
           batchSize, seqLen, d_k, d_v, numHeads);
    
    // Calculate memory sizes
    size_t qSize = batchSize * seqLen * d_k * sizeof(float);
    size_t kSize = batchSize * seqLen * d_k * sizeof(float);
    size_t vSize = batchSize * seqLen * d_v * sizeof(float);
    size_t outputSize = batchSize * seqLen * d_v * sizeof(float);
    
    // Half precision memory sizes
    size_t qSizeHalf = batchSize * seqLen * d_k * sizeof(half);
    size_t kSizeHalf = batchSize * seqLen * d_k * sizeof(half);
    size_t vSizeHalf = batchSize * seqLen * d_v * sizeof(half);
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output;
    half *d_Q_half, *d_K_half, *d_V_half;
    
    // Allocate host memory
    float *h_Q, *h_K, *h_V, *h_output;
    cudaMallocHost(&h_Q, qSize);
    cudaMallocHost(&h_K, kSize);
    cudaMallocHost(&h_V, vSize);
    cudaMallocHost(&h_output, outputSize);
    
    cudaMalloc(&d_Q, qSize);
    cudaMalloc(&d_K, kSize);
    cudaMalloc(&d_V, vSize);
    cudaMalloc(&d_output, outputSize);
    cudaMalloc(&d_Q_half, qSizeHalf);
    cudaMalloc(&d_K_half, kSizeHalf);
    cudaMalloc(&d_V_half, vSizeHalf);
    
    if (debug_mode) {
        printf("Memory allocation information:\n");
        printf("  Q size: %zu bytes (%zu elements)\n", qSize, qSize / sizeof(float));
        printf("  K size: %zu bytes (%zu elements)\n", kSize, kSize / sizeof(float));
        printf("  V size: %zu bytes (%zu elements)\n", vSize, vSize / sizeof(float));
        printf("  Output size: %zu bytes (%zu elements)\n", outputSize, outputSize / sizeof(float));
        printf("  Q half size: %zu bytes (%zu elements)\n", qSizeHalf, qSizeHalf / sizeof(half));
        printf("  K half size: %zu bytes (%zu elements)\n", kSizeHalf, kSizeHalf / sizeof(half));
        printf("  V half size: %zu bytes (%zu elements)\n", vSizeHalf, vSizeHalf / sizeof(half));
        printf("\n");
    }
    
    // Check memory allocation
    if (!h_Q || !h_K || !h_V || !h_output || !d_Q || !d_K || !d_V || !d_output || 
        !d_Q_half || !d_K_half || !d_V_half) {
        printf("Memory allocation failed!\n");
        return -1;
    }
    
    // Initialize data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < batchSize * seqLen * d_k; i++) {
        h_Q[i] = dis(gen);
        h_K[i] = dis(gen);
    }
    for (int i = 0; i < batchSize * seqLen * d_v; i++) {
        h_V[i] = dis(gen);
    }
    
    // Copy data to GPU
    cudaMemcpy(d_Q, h_Q, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, kSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vSize, cudaMemcpyHostToDevice);
    
    // Calculate conversion kernel configuration
    int blockSize = 256;
    size_t totalElements = batchSize * seqLen * d_k;
    size_t totalElementsV = batchSize * seqLen * d_v;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    int gridSizeV = (totalElementsV + blockSize - 1) / blockSize;
    
    if (debug_mode) {
        printf("Conversion kernel configuration:\n");
        printf("  Q/K grid size: %d, block size: %d, total elements: %zu\n", gridSize, blockSize, totalElements);
        printf("  V grid size: %d, block size: %d, total elements: %zu\n", gridSizeV, blockSize, totalElementsV);
        printf("\n");
    }
    
    // Launch conversion kernels
    convertFloatToHalfKernel<<<gridSize, blockSize>>>(d_Q, d_Q_half, totalElements);
    convertFloatToHalfKernel<<<gridSize, blockSize>>>(d_K, d_K_half, totalElements);
    convertFloatToHalfKernel<<<gridSizeV, blockSize>>>(d_V, d_V_half, totalElementsV);
    
    // Check if conversion was successful
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // Verify conversion results - check several sample values
    if (debug_mode) {
        half *h_Q_half_sample = new half[10];
        half *h_K_half_sample = new half[10];
        cudaMemcpy(h_Q_half_sample, d_Q_half, 10 * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_K_half_sample, d_K_half, 10 * sizeof(half), cudaMemcpyDeviceToHost);
        
        printf("Data conversion verification:\n");
        printf("  Original float Q[0]: %.6f -> half: %.6f\n", h_Q[0], __half2float(h_Q_half_sample[0]));
        printf("  Original float K[0]: %.6f -> half: %.6f\n", h_K[0], __half2float(h_K_half_sample[0]));
        printf("  Original float Q[1]: %.6f -> half: %.6f\n", h_Q[1], __half2float(h_Q_half_sample[1]));
        printf("  Original float K[1]: %.6f -> half: %.6f\n", h_K[1], __half2float(h_K_half_sample[1]));
        printf("\n");
        
        delete[] h_Q_half_sample;
        delete[] h_K_half_sample;
    }
    
    float scale = 1.0f / sqrtf(d_k);
    
    // Check if grid configuration is reasonable
    if (debug_mode) {
        printf("Checking grid configuration:\n");
        printf("  batchSize=%d, seqLen=%d, numHeads=%d\n", batchSize, seqLen, numHeads);
        printf("  d_k=%d, d_v=%d (different values to test indexing logic)\n", d_k, d_v);
        printf("  GPU max grid dimensions: x=%d, y=%d, z=%d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    
    // Fix grid configuration: ensure thread count matches data size
    dim3 gridDim(batchSize, seqLen, numHeads);
    dim3 blockDim(1);  // Each block only needs 1 thread
    
    if (debug_mode) {
        printf("Attention kernel configuration:\n");
        printf("  Grid: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
        printf("  Block: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
        printf("  Scale: %.6f\n", scale);
        printf("  Total threads: %d\n", gridDim.x * gridDim.y * gridDim.z * blockDim.x);
        
        // Verify index calculation
        printf("Index calculation verification:\n");
        printf("  Theoretical output array size: %d * %d * %d = %d\n", batchSize, seqLen, d_v, batchSize * seqLen * d_v);
        printf("  Actual output array size: %zu bytes / %zu = %zu elements\n", outputSize, sizeof(float), outputSize / sizeof(float));
        
        // Check several boundary case indices
        int max_seq_idx = seqLen - 1;
        int max_head_idx = numHeads - 1;
        int max_out_idx = batchSize * seqLen * d_v - 1;
        printf("  Max sequence index: %d\n", max_seq_idx);
        printf("  Max head index: %d\n", max_head_idx);
        printf("  Max output index: %d\n", max_out_idx);
        
        // Calculate indices for several key positions
        int idx_0_0_0 = 0 * seqLen * d_v + 0 * d_v;  // (0,0,0)
        int idx_0_63_3 = 0 * seqLen * d_v + 63 * d_v;  // (0,63,3)
        printf("  Index (0,0,0): %d\n", idx_0_0_0);
        printf("  Index (0,63,3): %d\n", idx_0_63_3);
        
        if (idx_0_63_3 >= batchSize * seqLen * d_v) {
            printf("  WARNING: Index (0,63,3) out of range!\n");
        }
        
        printf("\n");
    }
    
    // Check if grid configuration exceeds GPU limits
    if (gridDim.x > prop.maxGridSize[0] || gridDim.y > prop.maxGridSize[1] || gridDim.z > prop.maxGridSize[2]) {
        printf("ERROR: Grid configuration exceeds GPU limits!\n");
        return -1;
    }
    
    // Test standard precision version
    if (debug_mode) {
        printf("Starting standard precision version test...\n");
    }
    
    cudaMemset(d_output, 0, outputSize);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    standardPrecisionAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V,
                                                           d_output, scale,
                                                           batchSize, seqLen, d_k, d_v, numHeads);
    
    // Check kernel launch and execution
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Standard precision kernel launch failed: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Standard precision kernel execution failed: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    if (debug_mode) {
        printf("Standard precision kernel executed successfully, starting performance test...\n");
    }
    
    for (int i = 0; i < iterations; i++) {
        standardPrecisionAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V,
                                                               d_output, scale,
                                                               batchSize, seqLen, d_k, d_v, numHeads);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto standard_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Check if standard precision kernel was successful
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Standard precision kernel error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // Test mixed precision version
    if (debug_mode) {
        printf("Starting mixed precision version test...\n");
    }
    cudaMemset(d_output, 0, outputSize);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        mixedPrecisionAttentionKernel<<<gridDim, blockDim>>>(d_Q_half, d_K_half, d_V_half,
                                                            d_output, scale,
                                                            batchSize, seqLen, d_k, d_v, numHeads);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto mixed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Check if mixed precision kernel was successful
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Mixed precision kernel error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    if (debug_mode) {
        printf("Mixed precision kernel execution completed, time: %ld μs\n", mixed_time.count());
    }
    
    // Output results
    printf("\n=== Performance Test Results ===\n");
    printf("Performance comparison:\n");
    printf("  Standard precision (FP32): %ld μs (%.2f tokens/sec)\n", 
           standard_time.count() / iterations,
           (float)(batchSize * seqLen * iterations) / (standard_time.count() / 1000000.0f));
    
    printf("  Mixed precision (FP16): %ld μs (%.2f tokens/sec)\n", 
           mixed_time.count() / iterations,
           (float)(batchSize * seqLen * iterations) / (mixed_time.count() / 1000000.0f));
    
    float speedup = (float)standard_time.count() / mixed_time.count();
    printf("  Speedup: %.2fx\n", speedup);
    
    // Calculate memory savings
    size_t standardSize = qSize + kSize + vSize;
    size_t mixedSize = qSizeHalf + kSizeHalf + vSizeHalf;
    float memorySave = (1.0f - (float)mixedSize / standardSize) * 100.0f;
    printf("  Memory savings: %.1f%%\n", memorySave);
    
    if (debug_mode) {
        printf("\n=== Detailed Configuration Information ===\n");
        printf("Test configuration: batch=%d, seq_len=%d, d_k=%d, d_v=%d, heads=%d\n", 
               batchSize, seqLen, d_k, d_v, numHeads);
        printf("Iterations: %d\n", iterations);
        printf("GPU: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    }
    
    // Clean up resources
    cudaFreeHost(h_Q);
    cudaFreeHost(h_K);
    cudaFreeHost(h_V);
    cudaFreeHost(h_output);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    cudaFree(d_Q_half);
    cudaFree(d_K_half);
    cudaFree(d_V_half);
    
    printf("\nTest completed!\n");
    if (debug_mode) {
        printf("Use --help to see all available options\n");
    }
    return 0;
}
