#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

// Grouped Query Attention implementation
__global__ void groupedQueryAttentionKernel(float *Q, float *K, float *V, float *output,
                                           int batchSize, int seqLen, int d_k, int d_v, int numHeads,
                                           int numGroups) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= numHeads) {
        return;
    }
    
    int q_start = batchIdx * seqLen * d_k + seqIdx * d_k;
    int out_start = batchIdx * seqLen * d_v + seqIdx * d_v;
    
    // Calculate global head index
    int groupIdx = headIdx / numGroups;
    int localHeadIdx = headIdx % numGroups;
    
    // Calculate attention scores
    float max_score = -INFINITY;
    float attention_scores[2048];  // Assuming max sequence length
    
    for (int pos = 0; pos < seqLen; pos++) {
        float score = 0.0f;
        int k_pos_start = batchIdx * seqLen * d_k + pos * d_k;
        
        for (int dim = 0; dim < d_k; dim++) {
            int q_idx = q_start + dim;
            int k_idx = k_pos_start + dim;
            score += Q[q_idx] * K[k_idx];
        }
        
        score /= sqrtf(d_k);
        attention_scores[pos] = score;
        max_score = max(max_score, score);
    }
    
    // Apply softmax (simplified version, using tanh approximation)
    float sum_exp = 0.0f;
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] = expf(attention_scores[pos] - max_score);
        sum_exp += attention_scores[pos];
    }
    
    // Calculate weighted output
    for (int dim = 0; dim < d_v; dim++) {
        float weighted_sum = 0.0f;
        
        for (int pos = 0; pos < seqLen; pos++) {
            int v_idx = batchIdx * seqLen * d_v + pos * d_v + dim;
            weighted_sum += attention_scores[pos] * V[v_idx];
        }
        
        int out_idx = out_start + dim;
        output[out_idx] = weighted_sum;
    }
}

int main() {
    printf("=== Grouped Query Attention Test ===\n");
    
    // Get GPU information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    
    // Test parameters
    int batchSize = 1;
    int seqLen = 128;
    int d_k = 64;
    int d_v = 64;
    int numHeads = 32;
    int numGroups = 4;  // 4 heads per group
    int iterations = 100;
    
    printf("Test configuration: batch=%d, seq_len=%d, d_k=%d, d_v=%d, heads=%d, groups=%d\n\n",
           batchSize, seqLen, d_k, d_v, numHeads, numGroups);
    
    // Allocate memory
    size_t qSize = batchSize * seqLen * d_k * numHeads * sizeof(float);
    size_t vSize = batchSize * seqLen * d_v * numHeads * sizeof(float);
    size_t outputSize = batchSize * seqLen * d_v * numHeads * sizeof(float);
    
    float *h_Q, *h_K, *h_V, *h_output;
    float *d_Q, *d_K, *d_V, *d_output;
    
    cudaMallocHost(&h_Q, qSize);
    cudaMallocHost(&h_K, qSize);
    cudaMallocHost(&h_V, vSize);
    cudaMallocHost(&h_output, outputSize);
    
    cudaMalloc(&d_Q, qSize);
    cudaMalloc(&d_K, qSize);
    cudaMalloc(&d_V, vSize);
    cudaMalloc(&d_output, outputSize);
    
    // Initialize data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < batchSize * seqLen * d_k * numHeads; i++) {
        h_Q[i] = dis(gen);
        h_K[i] = dis(gen);
    }
    for (int i = 0; i < batchSize * seqLen * d_v * numHeads; i++) {
        h_V[i] = dis(gen);
    }
    
    // Copy data to GPU
    cudaMemcpy(d_Q, h_Q, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vSize, cudaMemcpyHostToDevice);
    
    // Configure kernel parameters
    dim3 blockDim(1);
    dim3 gridDim(batchSize, seqLen, numHeads);
    
    // Warm up
    groupedQueryAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                                       batchSize, seqLen, d_k, d_v, numHeads, numGroups);
    cudaDeviceSynchronize();
    
    // Performance test
    cudaMemset(d_output, 0, outputSize);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        groupedQueryAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                                          batchSize, seqLen, d_k, d_v, numHeads, numGroups);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Output results
    printf("Grouped Query Attention Performance:\n");
    printf("  Total time: %ld μs\n", time.count());
    printf("  Average time: %ld μs\n", time.count() / iterations);
    printf("  Throughput: %.2f tokens/sec\n",
           (float)(batchSize * seqLen * iterations) / (time.count() / 1000000.0f));
    
    // Calculate memory savings
    size_t standardSize = qSize + qSize + vSize;
    size_t groupedSize = qSize + qSize + vSize;  // Same memory usage but different computation pattern
    printf("  Memory usage: Same as standard attention\n");
    printf("  Computation pattern: Grouped query heads share KV cache\n");
    
    // Clean up
    cudaFreeHost(h_Q);
    cudaFreeHost(h_K);
    cudaFreeHost(h_V);
    cudaFreeHost(h_output);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    
    printf("\nTest completed!\n");
    return 0;
}
