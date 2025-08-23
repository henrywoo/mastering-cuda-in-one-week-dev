#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

// Flash Attention实现
#define BLOCK_SIZE 16

__global__ void flashAttentionKernel(float *Q, float *K, float *V, float *output,
                                    int batchSize, int seqLen, int d_k, int d_v,
                                    int blockSize) {
    __shared__ float s_Q[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_K[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_V[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_softmax[BLOCK_SIZE];
    
    // 将4维信息编码到3维中
    int batchIdx = blockIdx.x;
    int headIdx = blockIdx.y;
    int blockIdx_x = blockIdx.z / batchSize;  // 使用z维度的高位
    int blockIdx_y = blockIdx.z % batchSize;  // 使用z维度的低位
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 分块加载Q, K, V
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
    
    // 计算分块注意力分数
    float local_sum = 0.0f;
    float local_max = -INFINITY;
    
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float score = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            score += s_Q[ty][j] * s_K[i][j];
        }
        score /= sqrtf(d_k);
        
        // 在线softmax计算
        local_max = max(local_max, score);
        local_sum += expf(score - local_max);
    }
    
    // 协作计算全局最大值
    __shared__ float s_max[BLOCK_SIZE];
    
    s_max[ty] = local_max;
    __syncthreads();
    
    // 规约求全局最大值
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (ty < stride) {
            s_max[ty] = max(s_max[ty], s_max[ty + stride]);
        }
        __syncthreads();
    }
    
    float global_max = s_max[0];
    __syncthreads();
    
    // 重新计算softmax
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float score = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            score += s_Q[ty][j] * s_K[i][j];
        }
        score = (score - global_max) / sqrtf(d_k);
        s_softmax[i] = expf(score);
    }
    
    // 重新计算全局和用于归一化
    float global_sum = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        global_sum += s_softmax[i];
    }
    
    // 归一化softmax
    for (int i = 0; i < BLOCK_SIZE; i++) {
        s_softmax[i] /= global_sum;
    }
    
    // 计算输出
    for (int v = 0; v < BLOCK_SIZE; v++) {
        float weighted_sum = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            weighted_sum += s_softmax[i] * s_V[i][v];
        }
        
        int out_idx = batchIdx * seqLen * d_v + (blockIdx_x * BLOCK_SIZE + tx) * d_v + 
                      (headIdx * d_v + v);
        if (blockIdx_x * BLOCK_SIZE + tx < seqLen && headIdx * d_v + v < d_v) {
            output[out_idx] = weighted_sum;
        }
    }
}

int main() {
    printf("=== Flash Attention 测试 ===\n");
    
    // 获取GPU信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("\n");
    
    // 测试参数
    int batchSize = 1;
    int seqLen = 512;
    int d_k = 64;
    int d_v = 64;
    int numHeads = 8;
    int iterations = 100;
    
    printf("测试配置: batch=%d, seq_len=%d, d_k=%d, d_v=%d, heads=%d\n\n", 
           batchSize, seqLen, d_k, d_v, numHeads);
    
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
    
    // 复制数据到GPU
    cudaMemcpy(d_Q, h_Q, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, kSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vSize, cudaMemcpyHostToDevice);
    
    // 配置kernel参数 - 将4维信息编码到3维中
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    int blocksPerSeq = (seqLen + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDim(batchSize, numHeads, blocksPerSeq * blocksPerSeq);
    
    // 预热
    flashAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                               batchSize, seqLen, d_k, d_v, BLOCK_SIZE);
    cudaDeviceSynchronize();
    
    // 性能测试
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        flashAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_output,
                                                   batchSize, seqLen, d_k, d_v, BLOCK_SIZE);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 输出结果
    printf("Flash Attention 性能:\n");
    printf("  总时间: %ld μs\n", time.count());
    printf("  平均时间: %ld μs\n", time.count() / iterations);
    printf("  吞吐量: %.2f tokens/sec\n", 
           (float)(batchSize * seqLen * iterations) / (time.count() / 1000000.0f));
    
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
