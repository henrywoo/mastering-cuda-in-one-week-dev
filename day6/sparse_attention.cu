#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

// Sparse attention Kernel
__global__ void sparseAttentionKernel(float *Q, float *K, float *V, float *output,
                                     int *sparse_connections, int *connection_counts,
                                     int batchSize, int seqLen, int d_k, int d_v, int numHeads,
                                     int maxConnections) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= numHeads) {
        return;
    }
    
    int q_start = batchIdx * seqLen * d_k + seqIdx * d_k;
    int out_start = batchIdx * seqLen * d_v + seqIdx * d_v;
    
    // Get sparse connection information
    int connection_start = (batchIdx * seqLen + seqIdx) * maxConnections;
    int num_connections = connection_counts[batchIdx * seqLen + seqIdx];
    
    // Calculate sparse attention scores
    float max_score = -INFINITY;
    float attention_scores[64];  // Assuming max 64 connections
    
    for (int i = 0; i < num_connections; i++) {
        int pos = sparse_connections[connection_start + i];
        if (pos >= 0 && pos < seqLen) {
            float score = 0.0f;
            int k_pos_start = batchIdx * seqLen * d_k + pos * d_k;
            
            for (int dim = 0; dim < d_k; dim++) {
                int q_idx = q_start + dim;
                int k_idx = k_pos_start + dim;
                score += Q[q_idx] * K[k_idx];
            }
            
            score /= sqrtf(d_k);
            attention_scores[i] = score;
            max_score = max(max_score, score);
        }
    }
    
    // Apply softmax
    float sum_exp = 0.0f;
    for (int i = 0; i < num_connections; i++) {
        attention_scores[i] = expf(attention_scores[i] - max_score);
        sum_exp += attention_scores[i];
    }
    
    for (int i = 0; i < num_connections; i++) {
        attention_scores[i] /= sum_exp;
    }
    
    // Calculate weighted output
    for (int dim = 0; dim < d_v; dim++) {
        float weighted_sum = 0.0f;
        
        for (int i = 0; i < num_connections; i++) {
            int pos = sparse_connections[connection_start + i];
            if (pos >= 0 && pos < seqLen) {
                int v_idx = batchIdx * seqLen * d_v + pos * d_v + dim;
                weighted_sum += attention_scores[i] * V[v_idx];
            }
        }
        
        int out_idx = out_start + dim;
        output[out_idx] = weighted_sum;
    }
}

// Standard attention Kernel (for comparison)
__global__ void standardAttentionKernel(float *Q, float *K, float *V, float *output,
                                       int batchSize, int seqLen, int d_k, int d_v, int numHeads) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= numHeads) {
        return;
    }
    
    int q_start = batchIdx * seqLen * d_k + seqIdx * d_k;
    int out_start = batchIdx * seqLen * d_v + seqIdx * d_v;
    
    // Calculate attention scores for all positions
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
    
    // Apply softmax
    float sum_exp = 0.0f;
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] = expf(attention_scores[pos] - max_score);
        sum_exp += attention_scores[pos];
    }
    
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] /= sum_exp;
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
    printf("=== Sparse Attention Test ===\n");
    
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
    int numHeads = 8;
    int max_sparse_connections = 64;  // Each position connects to at most 64 other positions
    int iterations = 100;
    
    printf("Test configuration: batch=%d, seq_len=%d, d_k=%d, d_v=%d, heads=%d\n",
           batchSize, seqLen, d_k, d_v, numHeads);
    printf("Sparse connections: Each position connects to at most %d other positions\n\n", max_sparse_connections);
    
    // Allocate memory
    size_t qSize = batchSize * seqLen * d_k * numHeads * sizeof(float);
    size_t vSize = batchSize * seqLen * d_v * numHeads * sizeof(float);
    size_t outputSize = batchSize * seqLen * d_v * numHeads * sizeof(float);
    size_t sparseSize = batchSize * seqLen * max_sparse_connections * sizeof(int);
    size_t countSize = batchSize * seqLen * sizeof(int);
    
    float *h_Q, *h_K, *h_V, *h_output;
    int *h_sparse_connections, *h_connection_counts;
    float *d_Q, *d_K, *d_V, *d_output;
    int *d_sparse_connections, *d_connection_counts;
    
    cudaMallocHost(&h_Q, qSize);
    cudaMallocHost(&h_K, qSize);
    cudaMallocHost(&h_V, vSize);
    cudaMallocHost(&h_output, outputSize);
    cudaMallocHost(&h_sparse_connections, sparseSize);
    cudaMallocHost(&h_connection_counts, countSize);
    
    cudaMalloc(&d_Q, qSize);
    cudaMalloc(&d_K, qSize);
    cudaMalloc(&d_V, vSize);
    cudaMalloc(&d_output, outputSize);
    cudaMalloc(&d_sparse_connections, sparseSize);
    cudaMalloc(&d_connection_counts, countSize);
    
    // Initialize data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pos_dis(0, seqLen - 1);
    
    for (int i = 0; i < batchSize * seqLen * d_k * numHeads; i++) {
        h_Q[i] = dis(gen);
        h_K[i] = dis(gen);
    }
    for (int i = 0; i < batchSize * seqLen * d_v * numHeads; i++) {
        h_V[i] = dis(gen);
    }
    
    // Initialize sparse connections (random pattern)
    for (int b = 0; b < batchSize; b++) {
        for (int s = 0; s < seqLen; s++) {
            int base_idx = (b * seqLen + s) * max_sparse_connections;
            int num_conn = std::min(max_sparse_connections, seqLen / 2);
            h_connection_counts[b * seqLen + s] = num_conn;
            
            for (int c = 0; c < max_sparse_connections; c++) {
                if (c < num_conn) {
                    int target_pos;
                    do {
                        target_pos = pos_dis(gen);
                    } while (target_pos == s);
                    h_sparse_connections[base_idx + c] = target_pos;
                } else {
                    h_sparse_connections[base_idx + c] = -1;  // No connection
                }
            }
        }
    }
    
    // Copy data to device
    cudaMemcpy(d_Q, h_Q, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sparse_connections, h_sparse_connections, sparseSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_connection_counts, h_connection_counts, countSize, cudaMemcpyHostToDevice);
    
    // Kernel configuration
    dim3 blockDim(1);
    dim3 gridDim(batchSize, seqLen, numHeads);
    
    // Test sparse attention
    printf("Testing Sparse Attention...\n");
    cudaMemset(d_output, 0, outputSize);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        sparseAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                                    d_sparse_connections, d_connection_counts,
                                                    batchSize, seqLen, d_k, d_v, numHeads,
                                                    max_sparse_connections);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto sparse_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test standard attention
    printf("Testing Standard Attention...\n");
    cudaMemset(d_output, 0, outputSize);
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        standardAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                                      batchSize, seqLen, d_k, d_v, numHeads);
    }
    cudaDeviceSynchronize();
    
    end = std::chrono::high_resolution_clock::now();
    auto standard_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Results
    printf("\n=== Performance Results ===\n");
    printf("Sparse Attention: %ld μs (%.2f tokens/sec)\n",
           sparse_time.count() / iterations,
           (float)(batchSize * seqLen * iterations) / (sparse_time.count() / 1000000.0f));
    
    printf("Standard Attention: %ld μs (%.2f tokens/sec)\n",
           standard_time.count() / iterations,
           (float)(batchSize * seqLen * iterations) / (standard_time.count() / 1000000.0f));
    
    float speedup = (float)standard_time.count() / sparse_time.count();
    printf("Speedup: %.2fx\n", speedup);
    
    // Calculate memory savings
    size_t standardSize = qSize + qSize + vSize;
    size_t sparseSize_total = qSize + qSize + vSize + sparseSize + countSize;
    float memorySave = (1.0f - (float)sparseSize_total / standardSize) * 100.0f;
    printf("Memory savings: %.1f%%\n", memorySave);
    
    // Clean up
    cudaFreeHost(h_Q);
    cudaFreeHost(h_K);
    cudaFreeHost(h_V);
    cudaFreeHost(h_output);
    cudaFreeHost(h_sparse_connections);
    cudaFreeHost(h_connection_counts);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    cudaFree(d_sparse_connections);
    cudaFree(d_connection_counts);
    
    printf("\nTest completed!\n");
    return 0;
}
