# Day 5: Attention Mechanism and Transformer - Modern NLP CUDA Implementation

## Overview
Today we will learn the CUDA implementation of attention mechanisms and Transformers. This is the foundation of modern natural language processing, including the core concepts of models like BERT and GPT. We will deeply understand the computation process of self-attention mechanisms and various optimization techniques.

## Learning Objectives
- Understand the basic principles and mathematical formulas of attention mechanisms
- Master CUDA implementation of self-attention (Self-Attention)
- Learn to implement multi-head attention (Multi-Head Attention)
- Understand implementation of positional encoding and layer normalization
- Master complete implementation of Transformer architecture

## Attention Mechanism Basics

### 1. Mathematical Definition
The core of attention mechanisms is computing the relationship between Query (Q), Key (K), and Value (V):
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- Q: Query matrix (batch_size × seq_len × d_k)
- K: Key matrix (batch_size × seq_len × d_k)
- V: Value matrix (batch_size × seq_len × d_v)
- d_k: Key dimension

### 2. Computation Steps
1. **Compute attention scores**: S = QK^T
2. **Scale**: S' = S / √d_k
3. **Apply softmax**: A = softmax(S')
4. **Weighted sum**: Output = A × V

## Self-Attention Implementation

### 1. Basic Self-Attention Kernel
```cuda
__global__ void selfAttentionKernel(float *Q, float *K, float *V, float *output,
                                   int batchSize, int seqLen, int d_k, int d_v) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    int threadIdx = threadIdx.x;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= d_k) {
        return;
    }
    
    // Compute attention scores
    float attention_scores[seqLen];
    float max_score = -INFINITY;
    
    // Step 1: Compute QK^T and find maximum
    for (int j = 0; j < seqLen; j++) {
        float score = 0.0f;
        for (int k = 0; k < d_k; k++) {
            score += Q[batchIdx * seqLen * d_k + seqIdx * d_k + k] * 
                     K[batchIdx * seqLen * d_k + j * d_k + k];
        }
        attention_scores[j] = score / sqrtf(d_k);
        max_score = max(max_score, attention_scores[j]);
    }
    
    // Step 2: Apply softmax
    float sum_exp = 0.0f;
    for (int j = 0; j < seqLen; j++) {
        attention_scores[j] = expf(attention_scores[j] - max_score);
        sum_exp += attention_scores[j];
    }
    
    for (int j = 0; j < seqLen; j++) {
        attention_scores[j] /= sum_exp;
    }
    
    // Step 3: Compute weighted output
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

### 2. Optimized Version: Using Shared Memory
```cuda
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
    
    // Load Q, K, V data into shared memory
    // Compute attention scores with shared memory optimization
    // Apply softmax and compute weighted output
}
```

## Multi-Head Attention

### 1. Multi-Head Attention Implementation
```cuda
__global__ void multiHeadAttentionKernel(float *Q, float *K, float *V, float *output,
                                        int batchSize, int seqLen, int d_model, int numHeads) {
    int d_k = d_model / numHeads;
    int d_v = d_model / numHeads;
    
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= numHeads) {
        return;
    }
    
    // Compute attention for each head
    // Apply linear transformations
    // Concatenate results from all heads
}
```

### 2. Positional Encoding
```cuda
__global__ void positionalEncodingKernel(float *output, int seqLen, int d_model) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pos < seqLen && dim < d_model) {
        float pe;
        if (dim % 2 == 0) {
            pe = sinf(pos / powf(10000.0f, dim / (float)d_model));
        } else {
            pe = cosf(pos / powf(10000.0f, (dim - 1) / (float)d_model));
        }
        output[pos * d_model + dim] = pe;
    }
}
```

## Transformer Architecture

### 1. Transformer Block
```cuda
__global__ void transformerBlockKernel(float *input, float *output,
                                      float *qkv_weights, float *qkv_bias,
                                      float *proj_weights, float *proj_bias,
                                      float *ln1_weights, float *ln1_bias,
                                      float *ln2_weights, float *ln2_bias,
                                      int batchSize, int seqLen, int d_model) {
    // Multi-head attention
    // Add & Norm
    // Feed forward network
    // Add & Norm
}
```

### 2. Complete Transformer
```cuda
__global__ void transformerKernel(float *input, float *output,
                                 TransformerWeights *weights,
                                 int batchSize, int seqLen, int d_model, int numLayers) {
    // Process through all transformer layers
    // Apply final layer normalization
    // Generate output
}
```

## Performance Optimization

### 1. Memory Access Optimization
- Use shared memory for attention scores
- Optimize memory coalescing patterns
- Minimize global memory transactions

### 2. Computation Optimization
- Use fast math functions
- Optimize softmax computation
- Implement efficient matrix operations

### 3. Thread Block Configuration
- Choose optimal block dimensions
- Balance shared memory usage and occupancy
- Consider sequence length and model dimensions

## Quick Start

### 1. Compile Basic Version
```bash
nvcc -o self_attention self_attention.cu
```

### 2. Compile Multi-Head Version
```bash
nvcc -o multi_head_attention multi_head_attention.cu
```

### 3. Compile Transformer Block
```bash
nvcc -o transformer_block transformer_block.cu
```

### 4. Compile Complete Transformer
```bash
nvcc -o transformer transformer.cu
```

## Performance Analysis

### 1. Basic Profiling
```bash
nvprof ./self_attention
```

### 2. Memory Bandwidth Analysis
```bash
nvprof --metrics dram_read_throughput,dram_write_throughput ./self_attention
```

### 3. Kernel Analysis
```bash
nvprof --kernels selfAttentionKernel,multiHeadAttentionKernel ./multi_head_attention
```

## Summary

Today we have learned:
1. **Attention Basics**: Mathematical principles and implementation
2. **Self-Attention**: CUDA implementation of attention mechanisms
3. **Multi-Head Attention**: Parallel attention computation
4. **Transformer Architecture**: Complete neural network implementation
5. **Performance Optimization**: Memory access and computation optimization

**Key Concepts**:
- **Attention Scores**: Computing QK^T relationships
- **Softmax**: Numerical stability in attention computation
- **Multi-Head**: Parallel attention mechanisms
- **Positional Encoding**: Adding sequence position information

**Next Steps**:
- Experiment with different attention patterns
- Implement advanced transformer architectures
- Explore attention optimization techniques

## 📁 Quick File Links

**Main Files**:
- [README.md](README.md) - This tutorial file
- [self_attention.cu](self_attention.cu) - Basic self-attention implementation
- [multi_head_attention.cu](multi_head_attention.cu) - Multi-head attention
- [transformer_block.cu](transformer_block.cu) - Transformer block implementation
- [transformer.cu](transformer.cu) - Complete transformer architecture

**Compilation Commands**:
```bash
# Basic compilation
nvcc -o self_attention self_attention.cu

# With optimization
nvcc -O3 -arch=sm_89 -o multi_head_attention multi_head_attention.cu

# Transformer components
nvcc -o transformer_block transformer_block.cu
nvcc -o transformer transformer.cu
```
