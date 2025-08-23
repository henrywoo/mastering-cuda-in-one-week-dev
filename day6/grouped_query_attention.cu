#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

// Grouped Query Attention实现
__global__ void groupedQueryAttentionKernel(float *Q, float *K, float *V, float *output,
                                           int batchSize, int seqLen, int d_k, int d_v,
                                           int numHeads, int numGroups) {
    int batchIdx = blockIdx.x;
    int groupIdx = blockIdx.y;
    int seqIdx = blockIdx.z;
    int headIdx = threadIdx.x;
    
    if (batchIdx >= batchSize || groupIdx >= numGroups || seqIdx >= seqLen || 
        headIdx >= (numHeads / numGroups)) {
        return;
    }
    
    // 计算全局头部索引
    int globalHeadIdx = groupIdx * (numHeads / numGroups) + headIdx;
    
    // 计算注意力分数
    float attention_score = 0.0f;
    for (int k = 0; k < d_k; k++) {
        float q_val = Q[batchIdx * seqLen * numHeads * d_k + seqIdx * numHeads * d_k + 
                        globalHeadIdx * d_k + k];
        float k_val = K[batchIdx * seqLen * numGroups * d_k + seqIdx * numGroups * d_k + 
                        groupIdx * d_k + k];
        attention_score += q_val * k_val;
    }
    attention_score /= sqrtf(d_k);
    
    // 应用softmax (简化版本，使用tanh近似)
    attention_score = tanhf(attention_score);
    
    // 计算加权输出
    for (int v = 0; v < d_v; v++) {
        float v_val = V[batchIdx * seqLen * numGroups * d_v + seqIdx * numGroups * d_v + 
                        groupIdx * d_v + v];
        float weighted_val = attention_score * v_val;
        
        int out_idx = batchIdx * seqLen * numHeads * d_v + seqIdx * numHeads * d_v + 
                      globalHeadIdx * d_v + v;
        output[out_idx] = weighted_val;
    }
}

int main() {
    printf("=== Grouped Query Attention 测试 ===\n");
    
    // 获取GPU信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("\n");
    
    // 测试参数
    int batchSize = 4;
    int seqLen = 1024;
    int d_k = 128;
    int d_v = 128;
    int numHeads = 16;
    int numGroups = 4;  // 4个头部一组
    int iterations = 50;
    
    printf("测试配置: batch=%d, seq_len=%d, d_k=%d, d_v=%d, heads=%d, groups=%d\n\n", 
           batchSize, seqLen, d_k, d_v, numHeads, numGroups);
    
    // 分配内存
    size_t qSize = batchSize * seqLen * numHeads * d_k * sizeof(float);
    size_t kSize = batchSize * seqLen * numGroups * d_k * sizeof(float);
    size_t vSize = batchSize * seqLen * numGroups * d_v * sizeof(float);
    size_t outputSize = batchSize * seqLen * numHeads * d_v * sizeof(float);
    
    float *h_Q, *h_K, *h_V, *h_output;
    float *d_Q, *d_K, *d_V, *d_output;
    
    cudaMallocHost(&h_Q, qSize);
    cudaMallocHost(&h_K, kSize);
    cudaMallocHost(&h_V, vSize);
    cudaMallocHost(&h_output, outputSize);
    cudaMalloc(&d_Q, qSize);
    cudaMalloc(&d_K, kSize);
    cudaMalloc(&d_V, vSize);
    cudaMalloc(&d_output, outputSize);
    
    // 初始化数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < batchSize * seqLen * numHeads * d_k; i++) {
        h_Q[i] = dis(gen);
    }
    for (int i = 0; i < batchSize * seqLen * numGroups * d_k; i++) {
        h_K[i] = dis(gen);
    }
    for (int i = 0; i < batchSize * seqLen * numGroups * d_v; i++) {
        h_V[i] = dis(gen);
    }
    
    // 复制数据到GPU
    cudaMemcpy(d_Q, h_Q, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, kSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vSize, cudaMemcpyHostToDevice);
    
    // 配置kernel参数
    dim3 gridDim(batchSize, numGroups, seqLen);
    dim3 blockDim(numHeads / numGroups);
    
    // 预热
    groupedQueryAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                                       batchSize, seqLen, d_k, d_v,
                                                       numHeads, numGroups);
    cudaDeviceSynchronize();
    
    // 性能测试
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        groupedQueryAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                                          batchSize, seqLen, d_k, d_v,
                                                          numHeads, numGroups);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 输出结果
    printf("Grouped Query Attention 性能:\n");
    printf("  总时间: %ld μs\n", time.count());
    printf("  平均时间: %ld μs\n", time.count() / iterations);
    printf("  吞吐量: %.2f tokens/sec\n", 
           (float)(batchSize * seqLen * iterations) / (time.count() / 1000000.0f));
    
    // 计算内存节省
    size_t standardKSize = batchSize * seqLen * numHeads * d_k * sizeof(float);
    size_t standardVSize = batchSize * seqLen * numHeads * d_v * sizeof(float);
    size_t gqaKSize = batchSize * seqLen * numGroups * d_k * sizeof(float);
    size_t gqaVSize = batchSize * seqLen * numGroups * d_v * sizeof(float);
    
    float kMemorySave = (1.0f - (float)gqaKSize / standardKSize) * 100.0f;
    float vMemorySave = (1.0f - (float)gqaVSize / standardVSize) * 100.0f;
    
    printf("  内存节省: K矩阵 %.1f%%, V矩阵 %.1f%%\n", kMemorySave, vMemorySave);
    
    // 清理资源
    cudaFreeHost(h_Q);
    cudaFreeHost(h_K);
    cudaFreeHost(h_V);
    cudaFreeHost(h_output);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    
    printf("\n测试完成!\n");
    return 0;
}
