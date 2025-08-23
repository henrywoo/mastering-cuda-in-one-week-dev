# Day 5: 注意力机制和Transformer - 现代NLP的CUDA实现

## 概述
今天我们将学习注意力机制(Attention)和Transformer的CUDA实现。这是现代自然语言处理的基础，包括BERT、GPT等模型的核心理念。我们将深入理解自注意力机制的计算过程和各种优化技巧。

## 学习目标
- 理解注意力机制的基本原理和数学公式
- 掌握自注意力(Self-Attention)的CUDA实现
- 学会实现多头注意力(Multi-Head Attention)
- 理解位置编码和层归一化的实现
- 掌握Transformer架构的完整实现

## 注意力机制基础

### 1. 数学定义
注意力机制的核心是计算查询(Query)、键(Key)、值(Value)之间的关系：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

其中：
- Q: 查询矩阵 (batch_size × seq_len × d_k)
- K: 键矩阵 (batch_size × seq_len × d_k)
- V: 值矩阵 (batch_size × seq_len × d_v)
- d_k: 键的维度

### 2. 计算步骤
1. **计算注意力分数**: S = QK^T
2. **缩放**: S' = S / √d_k
3. **应用softmax**: A = softmax(S')
4. **加权求和**: Output = A × V

## 自注意力实现

### 1. 基础自注意力Kernel
```cpp
__global__ void selfAttentionKernel(float *Q, float *K, float *V, float *output,
                                   int batchSize, int seqLen, int d_k, int d_v) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    int threadIdx = threadIdx.x;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= d_k) {
        return;
    }
    
    // 计算注意力分数
    float attention_scores[seqLen];
    float max_score = -INFINITY;
    
    // 第一步：计算QK^T并找到最大值
    for (int j = 0; j < seqLen; j++) {
        float score = 0.0f;
        for (int k = 0; k < d_k; k++) {
            score += Q[batchIdx * seqLen * d_k + seqIdx * d_k + k] * 
                     K[batchIdx * seqLen * d_k + j * d_k + k];
        }
        attention_scores[j] = score / sqrtf(d_k);
        max_score = max(max_score, attention_scores[j]);
    }
    
    // 第二步：应用softmax
    float sum_exp = 0.0f;
    for (int j = 0; j < seqLen; j++) {
        attention_scores[j] = expf(attention_scores[j] - max_score);
        sum_exp += attention_scores[j];
    }
    
    for (int j = 0; j < seqLen; j++) {
        attention_scores[j] /= sum_exp;
    }
    
    // 第三步：计算加权输出
    for (int v = 0; v < d_v; v++) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < seqLen; j++) {
            weighted_sum += attention_scores[j] * 
                           V[batchIdx * seqLen * d_v + j * d_v + v];
        }
        output[batchIdx * seqLen * d_v + seqIdx * d_v + v] = weighted_sum;
    }
}
```

### 2. 优化版本：使用共享内存
```cpp
__global__ void selfAttentionSharedKernel(float *Q, float *K, float *V, float *output,
                                         int batchSize, int seqLen, int d_k, int d_v) {
    __shared__ float s_attention_scores[MAX_SEQ_LEN];
    __shared__ float s_temp_values[MAX_SEQ_LEN];
    
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    int tx = threadIdx.x;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= d_k) {
        return;
    }
    
    // 协作计算注意力分数
    for (int j = tx; j < seqLen; j += blockDim.x) {
        float score = 0.0f;
        for (int k = 0; k < d_k; k++) {
            score += Q[batchIdx * seqLen * d_k + seqIdx * d_k + k] * 
                     K[batchIdx * seqLen * d_k + j * d_k + k];
        }
        s_attention_scores[j] = score / sqrtf(d_k);
    }
    __syncthreads();
    
    // 协作计算softmax
    float max_score = -INFINITY;
    for (int j = tx; j < seqLen; j += blockDim.x) {
        max_score = max(max_score, s_attention_scores[j]);
    }
    
    // 规约求最大值
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            s_temp_values[tx] = max(s_temp_values[tx], s_temp_values[tx + stride]);
        }
        __syncthreads();
    }
    
    if (tx == 0) {
        max_score = s_temp_values[0];
    }
    __syncthreads();
    
    // 应用softmax
    for (int j = tx; j < seqLen; j += blockDim.x) {
        s_attention_scores[j] = expf(s_attention_scores[j] - max_score);
    }
    __syncthreads();
    
    // 计算输出
    for (int v = tx; v < d_v; v += blockDim.x) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < seqLen; j++) {
            weighted_sum += s_attention_scores[j] * 
                           V[batchIdx * seqLen * d_v + j * d_v + v];
        }
        output[batchIdx * seqLen * d_v + seqIdx * d_v + v] = weighted_sum;
    }
}
```

## 多头注意力实现

### 1. 多头注意力结构
```cpp
struct MultiHeadAttention {
    int numHeads;
    int d_model;
    int d_k;
    int d_v;
    
    float *W_q, *W_k, *W_v;  // 线性变换权重
    float *W_o;               // 输出投影权重
    float *b_q, *b_k, *b_v, *b_o;  // 偏置项
};

__global__ void multiHeadAttentionKernel(MultiHeadAttention *mha,
                                        float *input, float *output,
                                        int batchSize, int seqLen) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    int threadIdx = threadIdx.x;
    
    // 计算当前头部的偏移
    int headOffset = headIdx * mha->d_k;
    
    // 线性变换：Q, K, V
    float Q[MAX_D_K], K[MAX_D_K], V[MAX_D_V];
    
    // 计算Q
    for (int k = 0; k < mha->d_k; k++) {
        Q[k] = 0.0f;
        for (int i = 0; i < mha->d_model; i++) {
            Q[k] += input[batchIdx * seqLen * mha->d_model + seqIdx * mha->d_model + i] * 
                     mha->W_q[i * mha->d_k + k];
        }
        Q[k] += mha->b_q[headOffset + k];
    }
    
    // 类似地计算K和V
    // ...
    
    // 计算注意力
    // ...
    
    // 输出投影
    // ...
}
```

### 2. 批处理优化
```cpp
// 使用批处理矩阵乘法优化
void multiHeadAttentionOptimized(MultiHeadAttention *mha,
                                float *input, float *output,
                                int batchSize, int seqLen) {
    // 重塑输入为(batch_size * seq_len, d_model)
    // 使用cuBLAS进行批量矩阵乘法
    
    // 计算Q, K, V
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 d_k, batchSize * seqLen, d_model,
                 &alpha, mha->W_q, d_k,
                 input, d_model,
                 &beta, Q, d_k);
    
    // 类似地计算K和V
    // ...
    
    // 计算注意力分数
    // ...
}
```

## 位置编码实现

### 1. 正弦位置编码
```cpp
__global__ void positionalEncodingKernel(float *pos_enc, int seqLen, int d_model) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pos < seqLen && dim < d_model) {
        if (dim % 2 == 0) {
            pos_enc[pos * d_model + dim] = sinf(pos / powf(10000.0f, dim / d_model));
        } else {
            pos_enc[pos * d_model + dim] = cosf(pos / powf(10000.0f, (dim-1) / d_model));
        }
    }
}
```

### 2. 相对位置编码
```cpp
__global__ void relativePositionalEncodingKernel(float *rel_pos_enc,
                                                int seqLen, int d_k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < seqLen && j < seqLen) {
        int rel_pos = i - j;
        if (rel_pos >= 0 && rel_pos < seqLen) {
            // 计算相对位置编码
            // ...
        }
    }
}
```

## 层归一化实现

### 1. 层归一化Kernel
```cpp
__global__ void layerNormKernel(float *input, float *output,
                                float *gamma, float *beta,
                                int batchSize, int seqLen, int d_model) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int dimIdx = threadIdx.x;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || dimIdx >= d_model) {
        return;
    }
    
    // 计算均值
    float mean = 0.0f;
    for (int d = 0; d < d_model; d++) {
        mean += input[batchIdx * seqLen * d_model + seqIdx * d_model + d];
    }
    mean /= d_model;
    
    // 计算方差
    float variance = 0.0f;
    for (int d = 0; d < d_model; d++) {
        float diff = input[batchIdx * seqLen * d_model + seqIdx * d_model + d] - mean;
        variance += diff * diff;
    }
    variance /= d_model;
    
    // 应用归一化
    int idx = batchIdx * seqLen * d_model + seqIdx * d_model + dimIdx;
    float normalized = (input[idx] - mean) / sqrtf(variance + 1e-6f);
    output[idx] = gamma[dimIdx] * normalized + beta[dimIdx];
}
```

## Transformer架构实现

### 1. Transformer Block
```cpp
struct TransformerBlock {
    MultiHeadAttention self_attn;
    float *ffn_weights1, *ffn_weights2;
    float *ffn_bias1, *ffn_bias2;
    float *ln1_gamma, *ln1_beta;
    float *ln2_gamma, *ln2_beta;
    int d_model, d_ff;
};

__global__ void transformerBlockKernel(TransformerBlock *block,
                                      float *input, float *output,
                                      int batchSize, int seqLen) {
    // 自注意力 + 残差连接 + 层归一化
    // 前馈网络 + 残差连接 + 层归一化
    // ...
}
```

### 2. 完整Transformer
```cpp
class Transformer {
private:
    std::vector<TransformerBlock> layers;
    float *input_embedding;
    float *output_embedding;
    float *pos_encoding;
    
public:
    void forward(float *input, float *output, int batchSize, int seqLen);
    void backward(float *grad_output, float *grad_input);
};
```

## 性能优化技巧

### 1. 内存访问优化
- **数据布局**: 使用NHWC格式提高内存合并访问
- **共享内存**: 缓存频繁访问的数据
- **内存池**: 减少内存分配开销

### 2. 计算优化
- **融合操作**: 将多个操作合并到一个kernel中
- **向量化**: 使用向量化指令提高吞吐量
- **循环展开**: 减少循环开销

### 3. 并行化策略
- **序列并行**: 不同序列并行处理
- **头部并行**: 不同注意力头并行计算
- **时间步并行**: 序列内时间步并行

## 编译和运行

### 编译命令
```bash
nvcc -O3 -arch=sm_70 -lcublas -o transformer transformer.cu
```

### 运行命令
```bash
./transformer
```

## 性能基准测试

### 测试配置
- 序列长度: 512, 1024, 2048
- 模型维度: 512, 768, 1024
- 注意力头数: 8, 12, 16
- 批处理大小: 1, 4, 8, 16

### 性能指标
- **吞吐量**: 每秒处理的token数
- **延迟**: 单个序列的处理时间
- **内存使用**: GPU内存占用
- **计算效率**: FLOPS利用率

## 常见问题和解决方案

### 1. 数值稳定性
- 使用混合精度训练
- 梯度裁剪防止梯度爆炸
- 检查NaN/Inf值

### 2. 内存不足
- 使用梯度检查点
- 减少批处理大小
- 优化内存布局

### 3. 性能瓶颈
- 使用性能分析工具识别瓶颈
- 优化内存访问模式
- 调整线程块大小

## 下一步
明天我们将学习最新的LLM CUDA kernel定制优化技术，包括Flash Attention、Paged Attention等前沿技术。

## 练习
1. 实现不同注意力变体(Relative, Local, Sparse)
2. 添加dropout和残差连接
3. 实现完整的Transformer训练循环
4. 使用TensorRT优化推理性能

## 参考资料
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [CUDA Convolution Implementation](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- [cuDNN Library](https://docs.nvidia.com/deeplearning/cudnn/)
- [CNN Architecture Design](https://arxiv.org/abs/1512.03385)
- [Transformer Architecture Visualization](https://jalammar.github.io/illustrated-transformer/)
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)
- [Flash Attention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Paged Attention: From Interface to Implementation](https://arxiv.org/abs/2309.06180)
- [CUDA Performance Optimization](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [TensorRT Optimization Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
