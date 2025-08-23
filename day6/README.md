# Day 6: Latest LLM CUDA Kernel Customization and Optimization - Cutting-Edge Technology Practice

## Overview
Today we will dive deep into the latest LLM CUDA kernel customization and optimization techniques, including Flash Attention, Paged Attention, Grouped Query Attention, and other cutting-edge technologies. These optimization techniques can significantly improve the training and inference performance of large language models, making them a hot research direction in the current AI field.

## Learning Objectives
- Understand the principles and CUDA implementation of Flash Attention
- Master memory management optimization of Paged Attention
- Learn to implement Grouped Query Attention
- Understand sparse attention and sliding window attention
- Master the latest Tensor Core optimization techniques

## Flash Attention Implementation

### 1. Flash Attention Principles
Flash Attention reduces memory usage through tiled computation and online softmax, achieving O(N) memory complexity:

```
Core Algorithm Idea:
1. Process input sequences in tiles
2. Compute softmax online, avoiding storage of complete attention matrices
3. Use numerically stable algorithms to avoid overflow
```

### 2. Flash Attention CUDA Implementation
```cuda
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
    
    // Tiled loading of Q, K, V
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
    
    // Compute tiled attention scores
    float local_sum = 0.0f;
    float local_max = -INFINITY;
    
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float score = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            score += s_Q[ty][j] * s_K[i][j];
        }
        score /= sqrtf(d_k);
        
        // Online softmax computation
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
    
    // Apply softmax and compute output
    float global_max = s_max[0];
    float global_sum = 0.0f;
    
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float score = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            score += s_Q[ty][j] * s_K[i][j];
        }
        score = (score / sqrtf(d_k)) - global_max;
        s_softmax[i] = expf(score);
        global_sum += s_softmax[i];
    }
    
    // Normalize and compute weighted output
    for (int i = 0; i < BLOCK_SIZE; i++) {
        s_softmax[i] /= global_sum;
    }
    
    // Compute final output
    if (tx < BLOCK_SIZE && ty < BLOCK_SIZE) {
        float output_val = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            output_val += s_softmax[i] * s_V[i][ty];
        }
        
        int output_idx = batchIdx * seqLen * d_v + (blockIdx_x * BLOCK_SIZE + tx) * d_v + 
                         (headIdx * d_v + ty);
        if (blockIdx_x * BLOCK_SIZE + tx < seqLen && headIdx * d_v + ty < d_v) {
            output[output_idx] = output_val;
        }
    }
}
```

## Paged Attention Implementation

### 1. Paged Attention Principles
Paged Attention optimizes memory management by using virtual memory techniques similar to operating systems:

```
Key Features:
1. Virtual memory addressing for KV cache
2. Dynamic memory allocation and deallocation
3. Efficient memory reuse and garbage collection
4. Support for variable sequence lengths
```

### 2. Paged Attention CUDA Implementation
```cuda
__global__ void pagedAttentionKernel(float *Q, float *K_cache, float *V_cache,
                                    int *block_tables, int *block_tables_offsets,
                                    float *output, int batchSize, int seqLen,
                                    int d_k, int d_v, int blockSize) {
    __shared__ float s_Q[BLOCK_SIZE];
    __shared__ float s_K[BLOCK_SIZE];
    __shared__ float s_V[BLOCK_SIZE];
    
    int batchIdx = blockIdx.x;
    int headIdx = blockIdx.y;
    int seqIdx = blockIdx.z;
    int tx = threadIdx.x;
    
    if (seqIdx >= seqLen) return;
    
    // Load query for current position
    if (tx < d_k) {
        int q_idx = batchIdx * seqLen * d_k + seqIdx * d_k + tx;
        s_Q[tx] = Q[q_idx];
    }
    __syncthreads();
    
    float attention_scores[MAX_SEQ_LEN];
    float max_score = -INFINITY;
    
    // Compute attention scores using paged KV cache
    for (int block_idx = 0; block_idx < block_tables_offsets[batchIdx]; block_idx++) {
        int physical_block = block_tables[batchIdx * MAX_BLOCKS + block_idx];
        
        // Load K, V from physical block
        if (tx < d_k) {
            int k_idx = physical_block * blockSize * d_k + tx;
            s_K[tx] = K_cache[k_idx];
        }
        if (tx < d_v) {
            int v_idx = physical_block * blockSize * d_v + tx;
            s_V[tx] = V_cache[v_idx];
        }
        __syncthreads();
        
        // Compute attention scores for this block
        for (int pos = 0; pos < blockSize; pos++) {
            float score = 0.0f;
            for (int k = 0; k < d_k; k++) {
                score += s_Q[k] * s_K[pos * d_k + k];
            }
            score /= sqrtf(d_k);
            attention_scores[block_idx * blockSize + pos] = score;
            max_score = max(max_score, score);
        }
    }
    
    // Apply softmax
    float sum_exp = 0.0f;
    for (int i = 0; i < seqLen; i++) {
        attention_scores[i] = expf(attention_scores[i] - max_score);
        sum_exp += attention_scores[i];
    }
    
    for (int i = 0; i < seqLen; i++) {
        attention_scores[i] /= sum_exp;
    }
    
    // Compute weighted output
    if (tx < d_v) {
        float output_val = 0.0f;
        for (int block_idx = 0; block_idx < block_tables_offsets[batchIdx]; block_idx++) {
            int physical_block = block_tables[batchIdx * MAX_BLOCKS + block_idx];
            
            for (int pos = 0; pos < blockSize; pos++) {
                int v_idx = physical_block * blockSize * d_v + pos * d_v + tx;
                output_val += attention_scores[block_idx * blockSize + pos] * V_cache[v_idx];
            }
        }
        
        int output_idx = batchIdx * seqLen * d_v + seqIdx * d_v + tx;
        output[output_idx] = output_val;
    }
}
```

## Grouped Query Attention

### 1. Grouped Query Attention Principles
Grouped Query Attention reduces computational complexity by grouping query heads that share the same key-value pairs:

```
Benefits:
1. Reduced memory bandwidth requirements
2. Lower computational complexity
3. Maintained model quality
4. Better scalability for long sequences
```

### 2. Grouped Query Attention Implementation
```cuda
__global__ void groupedQueryAttentionKernel(float *Q, float *K, float *V, float *output,
                                           int batchSize, int seqLen, int d_k, int d_v,
                                           int numHeads, int numGroups) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    int tx = threadIdx.x;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= numHeads) {
        return;
    }
    
    int groupIdx = headIdx / (numHeads / numGroups);
    int d_k_per_group = d_k * (numHeads / numGroups);
    int d_v_per_group = d_v * (numHeads / numGroups);
    
    // Compute attention scores for grouped queries
    float attention_scores[seqLen];
    float max_score = -INFINITY;
    
    for (int j = 0; j < seqLen; j++) {
        float score = 0.0f;
        for (int k = 0; k < d_k; k++) {
            int q_idx = batchIdx * seqLen * d_k + seqIdx * d_k + k;
            int k_idx = batchIdx * seqLen * d_k_per_group + j * d_k_per_group + 
                       groupIdx * d_k + k;
            score += Q[q_idx] * K[k_idx];
        }
        attention_scores[j] = score / sqrtf(d_k);
        max_score = max(max_score, attention_scores[j]);
    }
    
    // Apply softmax
    float sum_exp = 0.0f;
    for (int j = 0; j < seqLen; j++) {
        attention_scores[j] = expf(attention_scores[j] - max_score);
        sum_exp += attention_scores[j];
    }
    
    for (int j = 0; j < seqLen; j++) {
        attention_scores[j] /= sum_exp;
    }
    
    // Compute weighted output
    for (int v = 0; v < d_v; v++) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < seqLen; j++) {
            int v_idx = batchIdx * seqLen * d_v_per_group + j * d_v_per_group + 
                       groupIdx * d_v + v;
            weighted_sum += attention_scores[j] * V[v_idx];
        }
        
        int output_idx = batchIdx * seqLen * d_v + seqIdx * d_v + v;
        output[output_idx] = weighted_sum;
    }
}
```

## Mixed Precision Attention

### 1. Mixed Precision Principles
Mixed precision attention uses FP16 for computation while maintaining FP32 for accumulation to balance performance and numerical stability:

```
Advantages:
1. Reduced memory usage (50% reduction)
2. Faster computation on modern GPUs
3. Better memory bandwidth utilization
4. Maintained numerical accuracy
```

### 2. Mixed Precision Implementation
```cuda
__global__ void mixedPrecisionAttentionKernel(half *Q, half *K, half *V, half *output,
                                             int batchSize, int seqLen, int d_k, int d_v,
                                             int numHeads) {
    int batchIdx = blockIdx.x;
    int seqIdx = blockIdx.y;
    int headIdx = blockIdx.z;
    int tx = threadIdx.x;
    
    if (batchIdx >= batchSize || seqIdx >= seqLen || headIdx >= numHeads) {
        return;
    }
    
    // Convert to float for computation
    float q_local[MAX_D_K];
    float k_local[MAX_D_K];
    float v_local[MAX_D_V];
    
    // Load and convert Q, K, V
    for (int k = 0; k < d_k; k++) {
        int q_idx = batchIdx * seqLen * d_k + seqIdx * d_k + k;
        q_local[k] = __half2float(Q[q_idx]);
    }
    
    // Compute attention scores in FP32
    float attention_scores[seqLen];
    float max_score = -INFINITY;
    
    for (int j = 0; j < seqLen; j++) {
        float score = 0.0f;
        for (int k = 0; k < d_k; k++) {
            int k_idx = batchIdx * seqLen * d_k + j * d_k + k;
            k_local[k] = __half2float(K[k_idx]);
            score += q_local[k] * k_local[k];
        }
        attention_scores[j] = score / sqrtf(d_k);
        max_score = max(max_score, attention_scores[j]);
    }
    
    // Apply softmax in FP32
    float sum_exp = 0.0f;
    for (int j = 0; j < seqLen; j++) {
        attention_scores[j] = expf(attention_scores[j] - max_score);
        sum_exp += attention_scores[j];
    }
    
    for (int j = 0; j < seqLen; j++) {
        attention_scores[j] /= sum_exp;
    }
    
    // Compute weighted output and convert back to FP16
    for (int v = 0; v < d_v; v++) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < seqLen; j++) {
            int v_idx = batchIdx * seqLen * d_v + j * d_v + v;
            v_local[v] = __half2float(V[v_idx]);
            weighted_sum += attention_scores[j] * v_local[v];
        }
        
        int output_idx = batchIdx * seqLen * d_v + seqIdx * d_v + v;
        output[output_idx] = __float2half(weighted_sum);
    }
}
```

## Performance Optimization Techniques

### 1. Memory Access Optimization
- Use shared memory for frequently accessed data
- Optimize memory coalescing patterns
- Implement efficient tiling strategies
- Minimize global memory transactions

### 2. Computation Optimization
- Use Tensor Core operations where possible
- Implement efficient softmax algorithms
- Optimize reduction operations
- Use fast math functions

### 3. Thread Block Configuration
- Choose optimal block dimensions for attention
- Balance shared memory usage and occupancy
- Consider sequence length and model dimensions
- Optimize for specific GPU architectures

## Quick Start

### 1. Compile Flash Attention
```bash
nvcc -o flash_attention flash_attention.cu
```

### 2. Compile Paged Attention
```bash
nvcc -o paged_attention paged_attention.cu
```

### 3. Compile Grouped Query Attention
```bash
nvcc -o grouped_query_attention grouped_query_attention.cu
```

### 4. Compile Mixed Precision Attention
```bash
nvcc -o mixed_precision_attention mixed_precision_attention.cu
```

## Performance Analysis

### 1. Basic Profiling
```bash
nvprof ./flash_attention
```

### 2. Memory Bandwidth Analysis
```bash
nvprof --metrics dram_read_throughput,dram_write_throughput ./flash_attention
```

### 3. Kernel Analysis
```bash
nvprof --kernels flashAttentionKernel,pagedAttentionKernel ./attention_benchmark
```

## Summary

Today we have learned:
1. **Flash Attention**: Tiled computation and online softmax
2. **Paged Attention**: Virtual memory management for KV cache
3. **Grouped Query Attention**: Shared key-value pairs optimization
4. **Mixed Precision**: FP16/FP32 hybrid computation
5. **Performance Optimization**: Advanced CUDA optimization techniques

**Key Concepts**:
- **Tiled Computation**: Break large matrices into manageable tiles
- **Online Softmax**: Compute softmax without storing full attention matrix
- **Virtual Memory**: Efficient memory management for variable sequences
- **Mixed Precision**: Balance performance and numerical accuracy

**Next Steps**:
- Experiment with different attention patterns
- Implement advanced optimization techniques
- Explore Tensor Core optimizations

## 📁 Quick File Links

**Main Files**:
- [README.md](README.md) - This tutorial file
- [flash_attention.cu](flash_attention.cu) - Flash Attention implementation
- [paged_attention.cu](paged_attention.cu) - Paged Attention implementation
- [grouped_query_attention.cu](grouped_query_attention.cu) - Grouped Query Attention
- [sparse_attention.cu](sparse_attention.cu) - Sparse Attention implementation
- [mixed_precision_attention.cu](mixed_precision_attention.cu) - Mixed Precision Attention

**Compilation Commands**:
```bash
# Flash Attention
nvcc -o flash_attention flash_attention.cu

# Paged Attention
nvcc -o paged_attention paged_attention.cu

# Grouped Query Attention
nvcc -o grouped_query_attention grouped_query_attention.cu

# Mixed Precision Attention
nvcc -o mixed_precision_attention mixed_precision_attention.cu

# With optimization
nvcc -O3 -arch=sm_89 -o flash_attention flash_attention.cu
```
