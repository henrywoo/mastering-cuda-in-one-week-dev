/*
 * 混合精度注意力机制实现
 * 
 * 编译选项（推荐使用compute-sanitizer来调试内存问题）：
 * 
 * 1. 普通编译：
 *    nvcc -O3 -o mixed_precision_attention mixed_precision_attention.cu
 * 
 * 2. 使用compute-sanitizer调试内存问题：
 *    nvcc -g -G -o mixed_precision_attention mixed_precision_attention.cu
 *    compute-sanitizer --tool memcheck ./mixed_precision_attention
 * 
 * 3. 使用compute-sanitizer检查未初始化内存：
 *    compute-sanitizer --tool initcheck ./mixed_precision_attention
 * 
 * 4. 使用compute-sanitizer检查竞争条件：
 *    compute-sanitizer --tool racecheck ./mixed_precision_attention
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>
#include <random>

// 常量定义
#define MAX_SEQ_LEN 2048  // 最大序列长度

// 完整的标准精度注意力 Kernel
__global__ void standardPrecisionAttentionKernel(float *Q, float *K, float *V, 
                                                float *output, float scale,
                                                int batchSize, int seqLen, int d_k, int d_v, int numHeads) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    
    // 边界检查
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= numHeads) {
        return;
    }
    
    // 计算当前序列位置在Q、K、V中的起始索引
    int q_start = batchIdx * seqLen * d_k + seqIdx * d_k;
    int out_start = batchIdx * seqLen * d_v + seqIdx * d_v;
    
    // 计算注意力分数：Q与所有序列位置的K的点积
    float max_score = -INFINITY;
    float attention_scores[MAX_SEQ_LEN];
    
    // 第一步：计算Q-K点积并找到最大值（用于数值稳定性）
    for (int pos = 0; pos < seqLen; pos++) {
        float score = 0.0f;
        int k_pos_start = batchIdx * seqLen * d_k + pos * d_k;
        
        // 计算Q[seqIdx]与K[pos]的点积
        for (int dim = 0; dim < d_k; dim++) {
            int q_idx = q_start + dim;
            int k_idx = k_pos_start + dim;
            
            if (q_idx < batchSize * seqLen * d_k && k_idx < batchSize * seqLen * d_k) {
                score += Q[q_idx] * K[k_idx];
            }
        }
        
        score *= scale;  // 应用缩放因子
        attention_scores[pos] = score;
        max_score = max(max_score, score);
    }
    
    // 第二步：计算softmax（数值稳定版本）
    float sum_exp = 0.0f;
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] = expf(attention_scores[pos] - max_score);
        sum_exp += attention_scores[pos];
    }
    
    // 归一化
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] /= sum_exp;
    }
    
    // 第三步：计算加权输出
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

// 完整的混合精度注意力 Kernel
__global__ void mixedPrecisionAttentionKernel(half *Q, half *K, half *V, 
                                             float *output, float scale,
                                             int batchSize, int seqLen, int d_k, int d_v, int numHeads) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    
    // 边界检查
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= numHeads) {
        return;
    }
    
    // 计算当前序列位置在Q、K、V中的起始索引
    int q_start = batchIdx * seqLen * d_k + seqIdx * d_k;
    int out_start = batchIdx * seqLen * d_v + seqIdx * d_v;
    
    // 计算注意力分数：Q与所有序列位置的K的点积
    float max_score = -INFINITY;
    float attention_scores[MAX_SEQ_LEN];
    
    // 第一步：计算Q-K点积并找到最大值（用于数值稳定性）
    for (int pos = 0; pos < seqLen; pos++) {
        float score = 0.0f;
        int k_pos_start = batchIdx * seqLen * d_k + pos * d_k;
        
        // 计算Q[seqIdx]与K[pos]的点积（使用FP16）
        for (int dim = 0; dim < d_k; dim++) {
            int q_idx = q_start + dim;
            int k_idx = k_pos_start + dim;
            
            if (q_idx < batchSize * seqLen * d_k && k_idx < batchSize * seqLen * d_k) {
                score += __half2float(Q[q_idx]) * __half2float(K[k_idx]);
            }
        }
        
        score *= scale;  // 应用缩放因子
        attention_scores[pos] = score;
        max_score = max(max_score, score);
    }
    
    // 第二步：计算softmax（数值稳定版本）
    float sum_exp = 0.0f;
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] = expf(attention_scores[pos] - max_score);
        sum_exp += attention_scores[pos];
    }
    
    // 归一化
    for (int pos = 0; pos < seqLen; pos++) {
        attention_scores[pos] /= sum_exp;
    }
    
    // 第三步：计算加权输出
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

// 数据转换kernel: float -> half
__global__ void convertFloatToHalfKernel(float *input, half *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

int main(int argc, char *argv[]) {
    // 解析命令行参数
    bool debug_mode = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-d") == 0) {
            debug_mode = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("用法: %s [选项]\n", argv[0]);
            printf("选项:\n");
            printf("  --debug, -d    启用调试模式，显示详细信息\n");
            printf("  --help, -h     显示此帮助信息\n");
            return 0;
        }
    }
    
    if (debug_mode) {
        printf("=== 调试模式已启用 ===\n");
    }
    
    // 初始化CUDA
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        printf("CUDA设备初始化失败: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("=== 混合精度注意力测试 ===\n");
    
    // 获取GPU信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    
    // 检查是否支持half精度
    if (prop.major < 6) {
        printf("警告: 当前GPU不支持half精度，将使用float精度模拟\n");
    }
    printf("\n");
    
    // 测试参数 - 使用不同的d_k和d_v值来测试索引逻辑
    int batchSize = 1;
    int seqLen = 64;
    int d_k = 32;    // 与d_v不同，测试索引计算
    int d_v = 64;    // 与d_k不同，测试索引计算
    int numHeads = 4;
    int iterations = 100;
    
    printf("测试配置: batch=%d, seq_len=%d, d_k=%d, d_v=%d, heads=%d\n\n", 
           batchSize, seqLen, d_k, d_v, numHeads);
    
    // 计算内存大小
    size_t qSize = batchSize * seqLen * d_k * sizeof(float);
    size_t kSize = batchSize * seqLen * d_k * sizeof(float);
    size_t vSize = batchSize * seqLen * d_v * sizeof(float);
    size_t outputSize = batchSize * seqLen * d_v * sizeof(float);
    
    // half精度内存大小
    size_t qSizeHalf = batchSize * seqLen * d_k * sizeof(half);
    size_t kSizeHalf = batchSize * seqLen * d_k * sizeof(half);
    size_t vSizeHalf = batchSize * seqLen * d_v * sizeof(half);
    
    // 分配设备内存
    float *d_Q, *d_K, *d_V, *d_output;
    half *d_Q_half, *d_K_half, *d_V_half;
    
    // 分配主机内存
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
        printf("内存分配信息:\n");
        printf("  Q size: %zu bytes (%zu elements)\n", qSize, qSize / sizeof(float));
        printf("  K size: %zu bytes (%zu elements)\n", kSize, kSize / sizeof(float));
        printf("  V size: %zu bytes (%zu elements)\n", vSize, vSize / sizeof(float));
        printf("  Output size: %zu bytes (%zu elements)\n", outputSize, outputSize / sizeof(float));
        printf("  Q half size: %zu bytes (%zu elements)\n", qSizeHalf, qSizeHalf / sizeof(half));
        printf("  K half size: %zu bytes (%zu elements)\n", kSizeHalf, kSizeHalf / sizeof(half));
        printf("  V half size: %zu bytes (%zu elements)\n", vSizeHalf, vSizeHalf / sizeof(half));
        printf("\n");
    }
    
    // 检查内存分配
    if (!h_Q || !h_K || !h_V || !h_output || !d_Q || !d_K || !d_V || !d_output || 
        !d_Q_half || !d_K_half || !d_V_half) {
        printf("内存分配失败!\n");
        return -1;
    }
    
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
    
    // 计算转换kernel的配置
    int blockSize = 256;
    size_t totalElements = batchSize * seqLen * d_k;
    size_t totalElementsV = batchSize * seqLen * d_v;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    int gridSizeV = (totalElementsV + blockSize - 1) / blockSize;
    
    if (debug_mode) {
        printf("转换kernel配置:\n");
        printf("  Q/K grid size: %d, block size: %d, total elements: %zu\n", gridSize, blockSize, totalElements);
        printf("  V grid size: %d, block size: %d, total elements: %zu\n", gridSizeV, blockSize, totalElementsV);
        printf("\n");
    }
    
    // 启动转换kernel
    convertFloatToHalfKernel<<<gridSize, blockSize>>>(d_Q, d_Q_half, totalElements);
    convertFloatToHalfKernel<<<gridSize, blockSize>>>(d_K, d_K_half, totalElements);
    convertFloatToHalfKernel<<<gridSizeV, blockSize>>>(d_V, d_V_half, totalElementsV);
    
    // 检查转换是否成功
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // 验证转换结果 - 检查几个样本值
    if (debug_mode) {
        half *h_Q_half_sample = new half[10];
        half *h_K_half_sample = new half[10];
        cudaMemcpy(h_Q_half_sample, d_Q_half, 10 * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_K_half_sample, d_K_half, 10 * sizeof(half), cudaMemcpyDeviceToHost);
        
        printf("数据转换验证:\n");
        printf("  原始float Q[0]: %.6f -> half: %.6f\n", h_Q[0], __half2float(h_Q_half_sample[0]));
        printf("  原始float K[0]: %.6f -> half: %.6f\n", h_K[0], __half2float(h_K_half_sample[0]));
        printf("  原始float Q[1]: %.6f -> half: %.6f\n", h_Q[1], __half2float(h_Q_half_sample[1]));
        printf("  原始float K[1]: %.6f -> half: %.6f\n", h_K[1], __half2float(h_K_half_sample[1]));
        printf("\n");
        
        delete[] h_Q_half_sample;
        delete[] h_K_half_sample;
    }
    
    float scale = 1.0f / sqrtf(d_k);
    
    // 检查grid配置是否合理
    if (debug_mode) {
        printf("检查grid配置:\n");
        printf("  batchSize=%d, seqLen=%d, numHeads=%d\n", batchSize, seqLen, numHeads);
        printf("  d_k=%d, d_v=%d (不同值测试索引逻辑)\n", d_k, d_v);
        printf("  GPU最大grid维度: x=%d, y=%d, z=%d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    
    // 修复grid配置：确保线程数量与数据大小匹配
    dim3 gridDim(batchSize, seqLen, numHeads);
    dim3 blockDim(1);  // 每个block只需要1个线程
    
    if (debug_mode) {
        printf("注意力kernel配置:\n");
        printf("  Grid: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
        printf("  Block: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
        printf("  Scale: %.6f\n", scale);
        printf("  总线程数: %d\n", gridDim.x * gridDim.y * gridDim.z * blockDim.x);
        
        // 验证索引计算
        printf("索引计算验证:\n");
        printf("  理论输出数组大小: %d * %d * %d = %d\n", batchSize, seqLen, d_v, batchSize * seqLen * d_v);
        printf("  实际输出数组大小: %zu bytes / %zu = %zu elements\n", outputSize, sizeof(float), outputSize / sizeof(float));
        
        // 检查几个边界情况的索引
        int max_seq_idx = seqLen - 1;
        int max_head_idx = numHeads - 1;
        int max_out_idx = batchSize * seqLen * d_v - 1;
        printf("  最大序列索引: %d\n", max_seq_idx);
        printf("  最大头部索引: %d\n", max_head_idx);
        printf("  最大输出索引: %d\n", max_out_idx);
        
        // 计算几个关键位置的索引
        int idx_0_0_0 = 0 * seqLen * d_v + 0 * d_v;  // (0,0,0)
        int idx_0_63_3 = 0 * seqLen * d_v + 63 * d_v;  // (0,63,3)
        printf("  索引(0,0,0): %d\n", idx_0_0_0);
        printf("  索引(0,63,3): %d\n", idx_0_63_3);
        
        if (idx_0_63_3 >= batchSize * seqLen * d_v) {
            printf("  WARNING: 索引(0,63,3)超出范围!\n");
        }
        
        printf("\n");
    }
    
    // 检查grid配置是否超出GPU限制
    if (gridDim.x > prop.maxGridSize[0] || gridDim.y > prop.maxGridSize[1] || gridDim.z > prop.maxGridSize[2]) {
        printf("ERROR: Grid配置超出GPU限制!\n");
        return -1;
    }
    
    // 测试标准精度版本
    if (debug_mode) {
        printf("开始测试标准精度版本...\n");
    }
    
    cudaMemset(d_output, 0, outputSize);
    auto start = std::chrono::high_resolution_clock::now();
    
    // 启动kernel
    standardPrecisionAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V,
                                                           d_output, scale,
                                                           batchSize, seqLen, d_k, d_v, numHeads);
    
    // 检查kernel启动和执行
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("标准精度kernel启动失败: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("标准精度kernel执行失败: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    if (debug_mode) {
        printf("标准精度kernel执行成功，开始性能测试...\n");
    }
    
    for (int i = 0; i < iterations; i++) {
        standardPrecisionAttentionKernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V,
                                                               d_output, scale,
                                                               batchSize, seqLen, d_k, d_v, numHeads);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto standard_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 检查标准精度kernel是否成功
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("标准精度kernel错误: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // 测试混合精度版本
    if (debug_mode) {
        printf("开始测试混合精度版本...\n");
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
    
    // 检查混合精度kernel是否成功
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("混合精度kernel错误: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    if (debug_mode) {
        printf("混合精度kernel执行完成，时间: %ld μs\n", mixed_time.count());
    }
    
    // 输出结果
    printf("\n=== 性能测试结果 ===\n");
    printf("性能对比:\n");
    printf("  标准精度 (FP32): %ld μs (%.2f tokens/sec)\n", 
           standard_time.count() / iterations,
           (float)(batchSize * seqLen * iterations) / (standard_time.count() / 1000000.0f));
    
    printf("  混合精度 (FP16): %ld μs (%.2f tokens/sec)\n", 
           mixed_time.count() / iterations,
           (float)(batchSize * seqLen * iterations) / (mixed_time.count() / 1000000.0f));
    
    float speedup = (float)standard_time.count() / mixed_time.count();
    printf("  加速比: %.2fx\n", speedup);
    
    // 计算内存节省
    size_t standardSize = qSize + kSize + vSize;
    size_t mixedSize = qSizeHalf + kSizeHalf + vSizeHalf;
    float memorySave = (1.0f - (float)mixedSize / standardSize) * 100.0f;
    printf("  内存节省: %.1f%%\n", memorySave);
    
    if (debug_mode) {
        printf("\n=== 详细配置信息 ===\n");
        printf("测试配置: batch=%d, seq_len=%d, d_k=%d, d_v=%d, heads=%d\n", 
               batchSize, seqLen, d_k, d_v, numHeads);
        printf("迭代次数: %d\n", iterations);
        printf("GPU: %s\n", prop.name);
        printf("计算能力: %d.%d\n", prop.major, prop.minor);
    }
    
    // 清理资源
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
    
    printf("\n测试完成!\n");
    if (debug_mode) {
        printf("使用 --help 查看所有可用选项\n");
    }
    return 0;
}
