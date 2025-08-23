#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

// Flash Attention implementation
#define BLOCK_SIZE 16

__global__ void flashAttentionKernel(float *Q, float *K, float *V, float *output,
                                    int batchSize, int seqLen, int d_k, int d_v,
                                    int blockSize) {
    __shared__ float s_Q[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_K[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_V[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_softmax[BLOCK_SIZE];
    
    // Encode 4D information into 3D
    int batchIdx = blockIdx.x;
    int headIdx = blockIdx.y;
    int blockIdx_x = blockIdx.z / batchSize;  // Use high bits of z dimension
    int blockIdx_y = blockIdx.z % batchSize;  // Use low bits of z dimension
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block-wise loading of Q, K, V
    if (tx < BLOCK_SIZE && ty < BLOCK_SIZE) {
        int q_idx = batchIdx * seqLen * d_k + (blockIdx_x * BLOCK_SIZE + tx) * d_k + 
                    (headIdx * d_k + ty);
        int k_idx = batchIdx * seqLen * d_k + (blockIdx_y * BLOCK_SIZE + tx) * d_k + 
                    (headIdx * d_k + ty);
        int v_idx = batchIdx * seqLen * d_v + (blockIdx_y * BLOCK_SIZE + tx) * d_v + 
                    (headIdx * d_v + ty);
        
        if (blockIdx_x * BLOCK_SIZE + tx < seqLen && headIdx * d_k + ty < d_k) {
            s_Q[ty][tx] = Q[q_idx];
        } else {
            s_Q[ty][tx] = 0.0f;
        }
        
        if (blockIdx_y * BLOCK_SIZE + tx < seqLen && headIdx * d_k + ty < d_k) {
            s_K[ty][tx] = K[k_idx];
        } else {
            s_K[ty][tx] = 0.0f;
        }
        
        if (blockIdx_y * BLOCK_SIZE + tx < seqLen && headIdx * d_v + ty < d_v) {
            s_V[ty][tx] = V[v_idx];
        } else {
            s_V[ty][tx] = 0.0f;
        }
    }
    __syncthreads();
    
    // Calculate block-wise attention scores
    float local_sum = 0.0f;
    float local_max = -INFINITY;
    
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float score = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            score += s_Q[ty][j] * s_K[i][j];
        }
        score /= sqrtf(d_k);
        
        // Online softmax calculation
        local_max = max(local_max, score);
        local_sum += expf(score - local_max);
    }
    
    // Cooperatively compute global maximum
    __shared__ float s_max[BLOCK_SIZE];
    
    s_max[ty] = local_max;
    __syncthreads();
    
    // Reduction to find global maximum
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (ty < stride) {
            s_max[ty] = max(s_max[ty], s_max[ty + stride]);
        }
        __syncthreads();
    }
    
    float global_max = s_max[0];
    __syncthreads();
    
    // Recalculate softmax
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float score = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            score += s_Q[ty][j] * s_K[i][j];
        }
        score = (score - global_max) / sqrtf(d_k);
        s_softmax[i] = expf(score);
    }
    
    // Recalculate global sum for normalization
    float global_sum = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        global_sum += s_softmax[i];
    }
    
    // Normalize softmax
    for (int i = 0; i < BLOCK_SIZE; i++) {
        s_softmax[i] /= global_sum;
    }
    __syncthreads();
    
    // Calculate weighted output
    float output_val = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        output_val += s_softmax[i] * s_V[i][tx];
    }
    
    // Store result
    if (tx < BLOCK_SIZE && ty < BLOCK_SIZE) {
        int out_idx = batchIdx * seqLen * d_v + (blockIdx_x * BLOCK_SIZE + tx) * d_v + 
                      (headIdx * d_v + ty);
        if (blockIdx_x * BLOCK_SIZE + tx < seqLen && headIdx * d_v + ty < d_v) {
            output[out_idx] = output_val;
        }
    }
}

// Standard attention kernel for comparison
__global__ void standardAttentionKernel(float *Q, float *K, float *V, float *output,
                                       int batchSize, int seqLen, int d_k, int d_v) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= 32) {
        return;
    }
    
    int q_start = batchIdx * seqLen * d_k + seqIdx * d_k;
    int out_start = batchIdx * seqLen * d_v + seqIdx * d_v;
    
    // Calculate attention scores
    float max_score = -INFINITY;
    float attention_scores[2048];  // Assuming max sequence length
    
    // Step 1: Calculate Q-K dot product and find maximum
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
    
    // Step 2: Calculate softmax (numerically stable)
    float sum_exp = 0.0f;
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] = expf(attention_scores[pos] - max_score);
        sum_exp += attention_scores[pos];
    }
    
    // Normalize
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] /= sum_exp;
    }
    
    // Step 3: Calculate weighted output
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
    const int batchSize = 1;
    const int seqLen = 64;
    const int d_k = 64;
    const int d_v = 64;
    const int numHeads = 32;
    const int iterations = 100;
    
    printf("=== Flash Attention Performance Test ===\n");
    printf("Configuration: batch=%d, seq_len=%d, d_k=%d, d_v=%d, heads=%d\n\n",
           batchSize, seqLen, d_k, d_v, numHeads);
    
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
    
    // Copy data to device
    cudaMemcpy(d_Q, h_Q, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vSize, cudaMemcpyHostToDevice);
    
    // Kernel configuration
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(batchSize, numHeads, (seqLen + BLOCK_SIZE - 1) / BLOCK_SIZE * batchSize);
    
    // Test Flash Attention
    printf("Testing Flash Attention...\n");
    cudaMemset(d_output, 0, outputSize);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        flashAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                                   batchSize, seqLen, d_k, d_v, BLOCK_SIZE);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto flash_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test Standard Attention
    printf("Testing Standard Attention...\n");
    cudaMemset(d_output, 0, outputSize);
    
    dim3 stdGridDim(batchSize, seqLen, numHeads);
    dim3 stdBlockDim(1);
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        standardAttentionKernel<<<stdGridDim, stdBlockDim>>>(d_Q, d_K, d_V, d_output,
                                                            batchSize, seqLen, d_k, d_v);
    }
    cudaDeviceSynchronize();
    
    end = std::chrono::high_resolution_clock::now();
    auto standard_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Results
    printf("\n=== Performance Results ===\n");
    printf("Flash Attention: %ld μs (%.2f tokens/sec)\n",
           flash_time.count() / iterations,
           (float)(batchSize * seqLen * iterations) / (flash_time.count() / 1000000.0f));
    
    printf("Standard Attention: %ld μs (%.2f tokens/sec)\n",
           standard_time.count() / iterations,
           (float)(batchSize * seqLen * iterations) / (standard_time.count() / 1000000.0f));
    
    float speedup = (float)standard_time.count() / flash_time.count();
    printf("Speedup: %.2fx\n", speedup);
    
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
