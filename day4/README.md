# Day 4: Convolutional Neural Networks (CNN) - CUDA Deep Learning Practice

## Overview
Today we will implement the core operations of Convolutional Neural Networks (CNN): convolution layers and forward propagation. This is one of the most fundamental and important operations in deep learning. We will learn how to efficiently implement convolution operations on GPUs. Additionally, we will explore NVIDIA's deep learning acceleration library cuDNN, which provides highly optimized convolution algorithm implementations.

## Learning Objectives
- Understand the mathematical principles and implementation of convolution operations
- Master CUDA implementation techniques for 2D convolution
- Learn to use shared memory to optimize convolution operations
- Understand performance characteristics of different convolution algorithms
- Master CNN forward propagation implementation
- Learn to use and optimize with cuDNN library

## Convolution Operation Basics

### 1. Mathematical Definition
For input feature map I (H×W×C) and convolution kernel K (Kh×Kw×C×F), the output feature map O (H'×W'×F) calculation:
```
O[h][w][f] = Σ(I[h+kh][w+kw][c] * K[kh][kw][c][f])
```

Where:
- H', W' = (H - Kh + 2*P) / S + 1 (considering padding P and stride S)
- kh, kw iterate over convolution kernel dimensions
- c iterates over input channels

### 2. Convolution Parameters
- **Kernel Size**: Convolution kernel size (e.g., 3×3, 5×5)
- **Stride**: Step size, controlling output feature map dimensions
- **Padding**: Padding, maintaining output dimensions
- **Channels**: Input and output channel counts

## NVIDIA cuDNN Library

### 1. cuDNN Introduction
cuDNN (CUDA Deep Neural Network library) is NVIDIA's deep learning acceleration library, containing highly optimized convolution, pooling, normalization, and other operations:

```cpp
#include <cudnn.h>

// Use cuDNN for convolution operations
void conv2dCuDNN(float *input, float *filter, float *output,
                  int batchSize, int inChannels, int inHeight, int inWidth,
                  int outChannels, int filterHeight, int filterWidth,
                  int padHeight, int padWidth, int strideHeight, int strideWidth) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    // Create descriptors
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    
    // Set input descriptor
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batchSize, inChannels, inHeight, inWidth);
    
    // Set filter descriptor
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               outChannels, inChannels, filterHeight, filterWidth);
    
    // Set convolution descriptor
    cudnnSetConvolution2dDescriptor(convDesc, padHeight, padWidth,
                                   strideHeight, strideWidth, 1, 1,
                                   CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    // Set output descriptor
    int outHeight, outWidth;
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc,
                                         &outHeight, &outWidth);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batchSize, outChannels, outHeight, outWidth);
    
    // Select optimal algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, outputDesc,
                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    
    // Get workspace size
    size_t workspaceSize = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc,
                                           outputDesc, algo, &workspaceSize);
    
    // Allocate workspace
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        cudaMalloc(&workspace, workspaceSize);
    }
    
    // Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, inputDesc, input, filterDesc, filter,
                           convDesc, algo, workspace, workspaceSize,
                           &beta, outputDesc, output);
    
    // Cleanup
    if (workspace) {
        cudaFree(workspace);
    }
    
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroy(cudnn);
}
```

**cuDNN Advantages:**
- Highly optimized for different GPU architectures
- Automatic algorithm selection
- Support for various data types and precisions
- Extensive testing and validation

## Basic Convolution Implementation

### 1. Naive Implementation
```cuda
__global__ void conv2dNaive(float *input, float *filter, float *output,
                            int batchSize, int inChannels, int inHeight, int inWidth,
                            int outChannels, int filterHeight, int filterWidth,
                            int padHeight, int padWidth, int strideHeight, int strideWidth) {
    int outH = (inHeight + 2 * padHeight - filterHeight) / strideHeight + 1;
    int outW = (inWidth + 2 * padWidth - filterWidth) / strideWidth + 1;
    
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int f = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (h < outH && w < outW && f < outChannels) {
        float sum = 0.0f;
        
        for (int c = 0; c < inChannels; c++) {
            for (int kh = 0; kh < filterHeight; kh++) {
                for (int kw = 0; kw < filterWidth; kw++) {
                    int inH = h * strideHeight + kh - padHeight;
                    int inW = w * strideWidth + kw - padWidth;
                    
                    if (inH >= 0 && inH < inHeight && inW >= 0 && inW < inWidth) {
                        int inputIdx = c * inHeight * inWidth + inH * inWidth + inW;
                        int filterIdx = f * inChannels * filterHeight * filterWidth + 
                                      c * filterHeight * filterWidth + kh * filterWidth + kw;
                        
                        sum += input[inputIdx] * filter[filterIdx];
                    }
                }
            }
        }
        
        int outputIdx = f * outH * outW + h * outW + w;
        output[outputIdx] = sum;
    }
}
```

### 2. Shared Memory Optimization
```cuda
__global__ void conv2dShared(float *input, float *filter, float *output,
                             int batchSize, int inChannels, int inHeight, int inWidth,
                             int outChannels, int filterHeight, int filterWidth,
                             int padHeight, int padWidth, int strideHeight, int strideWidth) {
    __shared__ float sInput[TILE_SIZE + 2][TILE_SIZE + 2];
    __shared__ float sFilter[FILTER_SIZE][FILTER_SIZE];
    
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int f = blockIdx.z * blockDim.z + threadIdx.z;
    
    int outH = (inHeight + 2 * padHeight - filterHeight) / strideHeight + 1;
    int outW = (inWidth + 2 * padWidth - filterWidth) / strideWidth + 1;
    
    if (h < outH && w < outW && f < outChannels) {
        float sum = 0.0f;
        
        // Load filter into shared memory
        if (threadIdx.x < filterWidth && threadIdx.y < filterHeight) {
            sFilter[threadIdx.y][threadIdx.x] = filter[f * inChannels * filterHeight * filterWidth + 
                                                      threadIdx.y * filterWidth + threadIdx.x];
        }
        
        for (int c = 0; c < inChannels; c++) {
            // Load input tile into shared memory
            int inH = h * strideHeight - padHeight;
            int inW = w * strideWidth - padWidth;
            
            if (inH + threadIdx.y >= 0 && inH + threadIdx.y < inHeight &&
                inW + threadIdx.x >= 0 && inW + threadIdx.x < inWidth) {
                sInput[threadIdx.y][threadIdx.x] = input[c * inHeight * inWidth + 
                                                        (inH + threadIdx.y) * inWidth + (inW + threadIdx.x)];
            } else {
                sInput[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute convolution
            for (int kh = 0; kh < filterHeight; kh++) {
                for (int kw = 0; kw < filterWidth; kw++) {
                    sum += sInput[threadIdx.y + kh][threadIdx.x + kw] * sFilter[kh][kw];
                }
            }
            __syncthreads();
        }
        
        int outputIdx = f * outH * outW + h * outW + w;
        output[outputIdx] = sum;
    }
}
```

## CNN Forward Propagation

### 1. Complete CNN Implementation
```cuda
// CNN forward propagation kernel
__global__ void cnnForward(float *input, float *conv1_weights, float *conv1_bias,
                           float *conv2_weights, float *conv2_bias,
                           float *fc1_weights, float *fc1_bias,
                           float *output, int batchSize) {
    // Implementation details for complete CNN forward pass
    // Including convolution, pooling, activation, and fully connected layers
}
```

### 2. Pooling Operations
```cuda
__global__ void maxPooling(float *input, float *output,
                           int batchSize, int channels, int height, int width,
                           int poolHeight, int poolWidth, int strideHeight, int strideWidth) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (h < height && w < width && c < channels) {
        float maxVal = -INFINITY;
        
        for (int ph = 0; ph < poolHeight; ph++) {
            for (int pw = 0; pw < poolWidth; pw++) {
                int inH = h * strideHeight + ph;
                int inW = w * strideWidth + pw;
                
                if (inH < height && inW < width) {
                    int inputIdx = c * height * width + inH * width + inW;
                    maxVal = max(maxVal, input[inputIdx]);
                }
            }
        }
        
        int outputIdx = c * height * width + h * width + w;
        output[outputIdx] = maxVal;
    }
}
```

## Performance Optimization

### 1. Memory Access Optimization
- Use shared memory for frequently accessed data
- Optimize memory coalescing patterns
- Minimize global memory transactions

### 2. Thread Block Configuration
- Choose optimal block dimensions for convolution
- Balance shared memory usage and occupancy
- Consider filter size and input dimensions

### 3. Algorithm Selection
- Use cuDNN for production applications
- Implement custom kernels for research
- Profile and compare different approaches

## Quick Start

### 1. Compile Basic Version
```bash
nvcc -o cnn_conv cnn_conv.cu
```

### 2. Compile Optimized Version
```bash
nvcc -O3 -arch=sm_89 -o cnn_conv_optimized cnn_conv_optimized.cu
```

### 3. Compile with cuDNN
```bash
nvcc -o cnn_conv_cudnn cnn_conv_cudnn.cu -lcudnn
```

### 4. Compile Complete CNN
```bash
nvcc -o cnn_forward cnn_forward.cu -lcudnn
```

## Performance Analysis

### 1. Basic Profiling
```bash
nvprof ./cnn_conv
```

### 2. Memory Bandwidth Analysis
```bash
nvprof --metrics dram_read_throughput,dram_write_throughput ./cnn_conv
```

### 3. Kernel Analysis
```bash
nvprof --kernels conv2dNaive,conv2dShared ./cnn_conv
```

## Summary

Today we have learned:
1. **Convolution Basics**: Mathematical principles and implementation
2. **CUDA Implementation**: Naive and optimized convolution kernels
3. **cuDNN Library**: Using optimized convolution algorithms
4. **CNN Forward Propagation**: Complete neural network implementation
5. **Performance Optimization**: Memory access and thread configuration

**Key Concepts**:
- **Tiled Convolution**: Break large inputs into manageable tiles
- **Shared Memory**: Cache frequently accessed data
- **Memory Coalescing**: Optimize memory access patterns
- **Library Usage**: Leverage cuDNN for production code

**Next Steps**:
- Experiment with different convolution algorithms
- Implement advanced CNN architectures
- Explore GPU memory optimization techniques

## 📁 Quick File Links

**Main Files**:
- [README.md](README.md) - This tutorial file
- [cnn_conv.cu](cnn_conv.cu) - Basic convolution implementation
- [cnn_conv_optimized.cu](cnn_conv_optimized.cu) - Shared memory optimized version
- [cnn_conv_cudnn.cu](cnn_conv_cudnn.cu) - cuDNN implementation
- [cnn_forward.cu](cnn_forward.cu) - Complete CNN forward propagation

**Compilation Commands**:
```bash
# Basic compilation
nvcc -o cnn_conv cnn_conv.cu

# With optimization
nvcc -O3 -arch=sm_89 -o cnn_conv_optimized cnn_conv_optimized.cu

# With cuDNN
nvcc -o cnn_conv_cudnn cnn_conv_cudnn.cu -lcudnn

# Complete CNN
nvcc -o cnn_forward cnn_forward.cu -lcudnn
```
