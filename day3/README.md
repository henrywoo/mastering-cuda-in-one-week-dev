# Day 3: Matrix Multiplication Optimization - CUDA Performance Tuning Practice

## Overview
Today we will learn matrix multiplication, which is a core operation in deep learning. We will start with simple implementations and gradually optimize to high-performance versions, learning various CUDA optimization techniques. Additionally, we will explore NVIDIA's official optimization libraries such as cuBLAS and CUTLASS, which have implemented highly optimized matrix multiplication algorithms.

## Learning Objectives
- Understand matrix multiplication algorithms and implementation
- Master CUDA shared memory usage
- Learn to use CUDA streams for asynchronous operations
- Understand the importance of memory coalescing access
- Master performance analysis and tuning techniques
- Learn to use NVIDIA official optimization libraries

## Matrix Multiplication Basics

### 1. Mathematical Definition
For matrices A (M×K) and B (K×N), the result matrix C (M×N) calculation:
```
C[i][j] = Σ(A[i][k] * B[k][j]) for k = 0 to K-1
```

### 2. Computational Complexity
- Time complexity: O(M×N×K)
- Space complexity: O(M×N + M×K + K×N)
- Memory access patterns greatly affect performance

## NVIDIA Official Optimization Libraries

### 1. cuBLAS (CUDA Basic Linear Algebra Subroutines)
cuBLAS is NVIDIA's basic linear algebra library, containing highly optimized matrix multiplication implementations:

```cpp
#include <cublas_v2.h>

// Use cuBLAS for matrix multiplication
void matrixMulCuBLAS(float *A, float *B, float *C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 M, N, K, &alpha, A, M, B, K, &beta, C, M);
    
    cublasDestroy(handle);
}

// Batch matrix multiplication
void batchMatrixMulCuBLAS(float *A[], float *B[], float *C[], 
                          int M, int N, int K, int batchSize) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Batch matrix multiplication
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       M, N, K, &alpha, A, M, B, K, &beta, C, M, batchSize);
    
    cublasDestroy(handle);
}
```

**cuBLAS Advantages:**
- Highly optimized for different GPU architectures
- Supports multiple data types and precisions
- Automatically selects optimal algorithms
- Fully tested and validated

### 2. CUTLASS (CUDA Templates for Linear Algebra Subroutines)
CUTLASS is NVIDIA's templated linear algebra library, allowing developers to customize and optimize matrix multiplication algorithms:

```cpp
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

// Use CUTLASS for matrix multiplication
void matrixMulCUTLASS(float *A, float *B, float *C, int M, int N, int K) {
    using ElementOutput = float;
    using ElementAccumulator = float;
    using ElementCompute = float;
    
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    
    using Gemm = cutlass::gemm::device::Gemm<
        ElementOutput, LayoutInputA,
        ElementOutput, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm70
    >;
    
    typename Gemm::Arguments arguments{
        {M, N, K},
        {A, M},
        {B, K},
        {C, M},
        {C, M},
        {alpha, beta}
    };
    
    Gemm gemm;
    gemm.initialize(arguments);
    gemm();
}
```

**CUTLASS Advantages:**
- Highly customizable algorithms
- Template-based design for flexibility
- Support for mixed precision
- Advanced optimization techniques

## Basic Matrix Multiplication Implementation

### 1. Naive Implementation
```cuda
__global__ void matrixMulNaive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### 2. Shared Memory Optimization
```cuda
__global__ void matrixMulShared(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; m++) {
        // Load tiles into shared memory
        if (row < M && m * TILE_SIZE + tx < K)
            sA[ty][tx] = A[row * K + m * TILE_SIZE + tx];
        else
            sA[ty][tx] = 0.0f;
            
        if (m * TILE_SIZE + ty < K && col < N)
            sB[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

## Performance Optimization Techniques

### 1. Memory Coalescing
- Ensure adjacent threads access adjacent memory addresses
- Use proper data layout (row-major vs column-major)
- Optimize memory access patterns

### 2. Shared Memory Usage
- Cache frequently accessed data
- Avoid bank conflicts
- Optimize tile sizes

### 3. Thread Block Configuration
- Choose optimal block dimensions
- Balance occupancy and resource usage
- Consider memory bandwidth constraints

## Quick Start

### 1. Compile Basic Version
```bash
nvcc -o matrix_mul matrix_mul.cu
```

### 2. Compile Optimized Version
```bash
nvcc -O3 -arch=sm_89 -o matrix_mul_optimized matrix_mul_optimized.cu
```

### 3. Compile with cuBLAS
```bash
nvcc -o matrix_mul_cublas matrix_mul_cublas.cu -lcublas
```

### 4. Compile with CUTLASS
```bash
nvcc -o matrix_mul_cutlass matrix_mul_cutlass.cu -I/path/to/cutlass/include
```

## Performance Analysis

### 1. Basic Profiling
```bash
nvprof ./matrix_mul
```

### 2. Detailed Metrics
```bash
nvprof --metrics all ./matrix_mul
```

### 3. Memory Bandwidth Analysis
```bash
nvprof --metrics dram_read_throughput,dram_write_throughput ./matrix_mul
```

## Summary

Today we have learned:
1. **Matrix Multiplication Basics**: Algorithm understanding and implementation
2. **Shared Memory Optimization**: Tiled computation for better performance
3. **NVIDIA Libraries**: cuBLAS and CUTLASS usage
4. **Performance Tuning**: Memory access optimization and thread configuration
5. **Performance Analysis**: Using profiling tools effectively

**Key Concepts**:
- **Tiled Computation**: Break large matrices into smaller tiles
- **Shared Memory**: Cache frequently accessed data
- **Memory Coalescing**: Optimize memory access patterns
- **Library Usage**: Leverage optimized implementations

**Next Steps**:
- Experiment with different tile sizes
- Compare performance with different libraries
- Explore advanced optimization techniques

## 📁 Quick File Links

**Main Files**:
- [README.md](README.md) - This tutorial file
- [matrix_mul.cu](matrix_mul.cu) - Basic matrix multiplication
- [matrix_mul_optimized.cu](matrix_mul_optimized.cu) - Shared memory optimized version
- [matrix_mul_cublas.cu](matrix_mul_cublas.cu) - cuBLAS implementation
- [matrix_mul_cutlass.cu](matrix_mul_cutlass.cu) - CUTLASS implementation

**Compilation Commands**:
```bash
# Basic compilation
nvcc -o matrix_mul matrix_mul.cu

# With optimization
nvcc -O3 -arch=sm_89 -o matrix_mul_optimized matrix_mul_optimized.cu

# With cuBLAS
nvcc -o matrix_mul_cublas matrix_mul_cublas.cu -lcublas

# With CUTLASS
nvcc -o matrix_mul_cutlass matrix_mul_cutlass.cu -I/path/to/cutlass/include
```
