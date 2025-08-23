#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

// 稀疏注意力 Kernel
__global__ void sparseAttentionKernel(float *Q, float *K, float *V, float *output,
                                     int *sparse_indices, int *sparse_offsets,
                                     int batchSize, int seqLen, int d_k, int d_v,
                                     int max_sparse_connections) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    int tx = threadIdx.x;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= d_k) {
        return;
    }
    
    // 获取稀疏连接信息
    int start_idx = sparse_offsets[batchIdx * seqLen + seqIdx];
    int end_idx = sparse_offsets[batchIdx * seqLen + seqIdx + 1];
    int num_connections = end_idx - start_idx;
    
    if (num_connections > max_sparse_connections) {
        num_connections = max_sparse_connections;
    }
    
    // 计算稀疏注意力分数
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    for (int i = 0; i < num_connections; i++) {
        int target_idx = sparse_indices[start_idx + i];
        if (target_idx < 0 || target_idx >= seqLen) continue;
        
        float score = 0.0f;
        for (int k = 0; k < d_k; k++) {
            float q_val = Q[batchIdx * seqLen * d_k + seqIdx * d_k + k];
            float k_val = K[batchIdx * seqLen * d_k + target_idx * d_k + k];
            score += q_val * k_val;
        }
        score /= sqrtf(d_k);
        
        max_score = max(max_score, score);
        sum_exp += expf(score - max_score);
    }
    
    // 应用softmax
    for (int i = 0; i < num_connections; i++) {
        int target_idx = sparse_indices[start_idx + i];
        if (target_idx < 0 || target_idx >= seqLen) continue;
        
        float score = 0.0f;
        for (int k = 0; k < d_k; k++) {
            float q_val = Q[batchIdx * seqLen * d_k + seqIdx * d_k + k];
            float k_val = K[batchIdx * seqLen * d_k + target_idx * d_k + k];
            score += q_val * k_val;
        }
        score = (score - max_score) / sqrtf(d_k);
        float attention_weight = expf(score) / sum_exp;
        
        // 计算加权输出
        for (int v = 0; v < d_v; v++) {
            float v_val = V[batchIdx * seqLen * d_v + target_idx * d_v + v];
            float weighted_val = attention_weight * v_val;
            
            int out_idx = batchIdx * seqLen * d_v + seqIdx * d_v + v;
            atomicAdd(&output[out_idx], weighted_val);
        }
    }
}

// 标准注意力 Kernel (用于对比)
__global__ void standardAttentionKernel(float *Q, float *K, float *V, float *output,
                                       int batchSize, int seqLen, int d_k, int d_v) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    int tx = threadIdx.x;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= d_k) {
        return;
    }
    
    // 计算所有位置的注意力分数
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    for (int j = 0; j < seqLen; j++) {
        float score = 0.0f;
        for (int k = 0; k < d_k; k++) {
            float q_val = Q[batchIdx * seqLen * d_k + seqIdx * d_k + k];
            float k_val = K[batchIdx * seqLen * d_k + j * d_k + k];
            score += q_val * k_val;
        }
        score /= sqrtf(d_k);
        
        max_score = max(max_score, score);
        sum_exp += expf(score - max_score);
    }
    
    // 应用softmax
    for (int j = 0; j < seqLen; j++) {
        float score = 0.0f;
        for (int k = 0; k < d_k; k++) {
            float q_val = Q[batchIdx * seqLen * d_k + seqIdx * d_k + k];
            float k_val = K[batchIdx * seqLen * d_k + j * d_k + k];
            score += q_val * k_val;
        }
        score = (score - max_score) / sqrtf(d_k);
        float attention_weight = expf(score) / sum_exp;
        
        // 计算加权输出
        for (int v = 0; v < d_v; v++) {
            float v_val = V[batchIdx * seqLen * d_v + j * d_v + v];
            float weighted_val = attention_weight * v_val;
            
            int out_idx = batchIdx * seqLen * d_v + seqIdx * d_v + v;
            output[out_idx] += weighted_val;
        }
    }
}

int main() {
    printf("=== 稀疏注意力测试 ===\n");
    
    // 获取GPU信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("\n");
    
    // 测试参数
    int batchSize = 2;
    int seqLen = 1024;
    int d_k = 128;
    int d_v = 128;
    int numHeads = 16;
    int max_sparse_connections = 64;  // 每个位置最多连接64个其他位置
    int iterations = 30;
    
    printf("测试配置: batch=%d, seq_len=%d, d_k=%d, d_v=%d, heads=%d\n", 
           batchSize, seqLen, d_k, d_v, numHeads);
    printf("稀疏连接: 每个位置最多连接 %d 个其他位置\n\n", max_sparse_connections);
    
    // 分配内存
    size_t qSize = batchSize * seqLen * d_k * sizeof(float);
    size_t kSize = batchSize * seqLen * d_k * sizeof(float);
    size_t vSize = batchSize * seqLen * d_v * sizeof(float);
    size_t outputSize = batchSize * seqLen * d_v * sizeof(float);
    
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
    
    for (int i = 0; i < batchSize * seqLen * d_k; i++) {
        h_Q[i] = dis(gen);
        h_K[i] = dis(gen);
    }
    for (int i = 0; i < batchSize * seqLen * d_v; i++) {
        h_V[i] = dis(gen);
    }
    
    // 创建稀疏连接模式 (局部注意力 + 随机连接)
    std::vector<int> sparse_indices;
    std::vector<int> sparse_offsets(seqLen + 1, 0);
    
    for (int i = 0; i < seqLen; i++) {
        std::vector<int> connections;
        
        // 添加局部连接 (前后各32个位置)
        for (int j = max(0, i - 32); j <= min(seqLen - 1, i + 32); j++) {
            if (j != i) {
                connections.push_back(j);
            }
        }
        
        // 添加一些随机连接
        std::uniform_int_distribution<int> rand_dis(0, seqLen - 1);
        for (int j = 0; j < 16; j++) {
            int rand_pos = rand_dis(gen);
            if (rand_pos != i && 
                std::find(connections.begin(), connections.end(), rand_pos) == connections.end()) {
                connections.push_back(rand_pos);
            }
        }
        
        // 限制连接数量
        if (connections.size() > max_sparse_connections) {
            connections.resize(max_sparse_connections);
        }
        
        sparse_offsets[i + 1] = sparse_offsets[i] + connections.size();
        for (int j : connections) {
            sparse_indices.push_back(j);
        }
    }
    
    int *d_sparse_indices, *d_sparse_offsets;
    cudaMalloc(&d_sparse_indices, sparse_indices.size() * sizeof(int));
    cudaMalloc(&d_sparse_offsets, sparse_offsets.size() * sizeof(int));
    
    cudaMemcpy(d_sparse_indices, sparse_indices.data(), 
               sparse_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sparse_offsets, sparse_offsets.data(), 
               sparse_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // 复制数据到GPU
    cudaMemcpy(d_Q, h_Q, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, kSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vSize, cudaMemcpyHostToDevice);
    
    // 配置kernel参数
    dim3 gridDim(batchSize, seqLen, numHeads);
    dim3 blockDim(256);
    
    // 测试标准注意力版本
    cudaMemset(d_output, 0, outputSize);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        standardAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                                      batchSize, seqLen, d_k, d_v);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto standard_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试稀疏注意力版本
    cudaMemset(d_output, 0, outputSize);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        sparseAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                                    d_sparse_indices, d_sparse_offsets,
                                                    batchSize, seqLen, d_k, d_v,
                                                    max_sparse_connections);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto sparse_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 输出结果
    printf("性能对比:\n");
    printf("  标准注意力: %ld μs (%.2f tokens/sec)\n", 
           standard_time.count() / iterations,
           (float)(batchSize * seqLen * iterations) / (standard_time.count() / 1000000.0f));
    
    printf("  稀疏注意力: %ld μs (%.2f tokens/sec)\n", 
           sparse_time.count() / iterations,
           (float)(batchSize * seqLen * iterations) / (sparse_time.count() / 1000000.0f));
    
    float speedup = (float)standard_time.count() / sparse_time.count();
    printf("  加速比: %.2fx\n", speedup);
    
    // 计算计算量节省
    int standard_ops = seqLen * seqLen * d_k * d_v;
    int sparse_ops = sparse_indices.size() * d_k * d_v;
    float compute_save = (1.0f - (float)sparse_ops / standard_ops) * 100.0f;
    printf("  计算量节省: %.1f%%\n", compute_save);
    
    // 清理资源
    cudaFreeHost(h_Q);
    cudaFreeHost(h_K);
    cudaFreeHost(h_V);
    cudaFreeHost(h_output);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    cudaFree(d_sparse_indices);
    cudaFree(d_sparse_offsets);
    
    printf("\n测试完成!\n");
    return 0;
}
