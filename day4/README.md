# Day 4: 卷积神经网络(CNN) - CUDA深度学习实战

## 概述
今天我们将实现卷积神经网络(CNN)的核心操作：卷积层和前向传播。这是深度学习中最基础也是最重要的操作之一，我们将学习如何高效地在GPU上实现卷积运算。同时，我们还将了解NVIDIA提供的深度学习加速库cuDNN，它提供了高度优化的卷积算法实现。

## 学习目标
- 理解卷积运算的数学原理和实现
- 掌握2D卷积的CUDA实现技巧
- 学会使用共享内存优化卷积操作
- 理解不同卷积算法的性能特点
- 掌握CNN前向传播的实现
- 了解cuDNN库的使用和优化

## 卷积运算基础

### 1. 数学定义
对于输入特征图 I (H×W×C) 和卷积核 K (Kh×Kw×C×F)，输出特征图 O (H'×W'×F) 的计算：
```
O[h][w][f] = Σ(I[h+kh][w+kw][c] * K[kh][kw][c][f])
```

其中：
- H', W' = (H - Kh + 2*P) / S + 1 (考虑填充P和步长S)
- kh, kw 遍历卷积核的尺寸
- c 遍历输入通道数

### 2. 卷积参数
- **Kernel Size**: 卷积核大小 (如3×3, 5×5)
- **Stride**: 步长，控制输出特征图的尺寸
- **Padding**: 填充，保持输出尺寸
- **Channels**: 输入和输出通道数

## NVIDIA cuDNN库

### 1. cuDNN简介
cuDNN (CUDA Deep Neural Network library) 是NVIDIA提供的深度学习加速库，包含高度优化的卷积、池化、归一化等操作：

```cpp
#include <cudnn.h>

// 使用cuDNN进行卷积运算
void conv2dCuDNN(float *input, float *filter, float *output,
                  int batchSize, int inChannels, int inHeight, int inWidth,
                  int outChannels, int filterHeight, int filterWidth,
                  int padHeight, int padWidth, int strideHeight, int strideWidth) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    // 创建描述符
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    
    // 设置输入描述符
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batchSize, inChannels, inHeight, inWidth);
    
    // 设置滤波器描述符
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               outChannels, inChannels, filterHeight, filterWidth);
    
    // 设置卷积描述符
    cudnnSetConvolution2dDescriptor(convDesc, padHeight, padWidth,
                                   strideHeight, strideWidth, 1, 1,
                                   CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    // 设置输出描述符
    int outHeight, outWidth;
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc,
                                         &outHeight, &outWidth);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batchSize, outChannels, outHeight, outWidth);
    
    // 选择最优算法
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, outputDesc,
                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    
    // 获取工作空间大小
    size_t workspaceSize = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc,
                                           outputDesc, algo, &workspaceSize);
    
    // 分配工作空间
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        cudaMalloc(&workspace, workspaceSize);
    }
    
    // 执行卷积
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, inputDesc, input, filterDesc, filter,
                           convDesc, algo, workspace, workspaceSize,
                           &beta, outputDesc, output);
    
    // 清理资源
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);
}
```

### 2. cuDNN优化特性
- **算法自动选择**: 自动选择最优的卷积算法
- **内存优化**: 最小化工作空间使用
- **多精度支持**: 支持FP16、FP32、FP64等精度
- **Tensor Core支持**: 在支持的GPU上自动使用Tensor Core

### 3. cuDNN卷积算法
```cpp
// 不同的卷积算法
enum cudnnConvolutionFwdAlgo_t {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,      // 隐式GEMM
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, // 预编译GEMM
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,               // 显式GEMM
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,             // 直接卷积
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,                // FFT卷积
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,         // 分块FFT卷积
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,           // Winograd卷积
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED   // 非融合Winograd卷积
};

// 选择算法策略
void selectOptimalAlgorithm(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inputDesc,
                           cudnnFilterDescriptor_t filterDesc, cudnnConvolutionDescriptor_t convDesc,
                           cudnnTensorDescriptor_t outputDesc) {
    // 获取所有可用算法
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[8];
    cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, outputDesc,
                                        8, &returnedAlgoCount, perfResults);
    
    // 选择最优算法
    cudnnConvolutionFwdAlgo_t bestAlgo = perfResults[0].algo;
    printf("Best algorithm: %d, time: %f ms, memory: %zu bytes\n",
           bestAlgo, perfResults[0].time, perfResults[0].memory);
    
    // 打印所有算法性能
    for (int i = 0; i < returnedAlgoCount; i++) {
        printf("Algorithm %d: time=%f ms, memory=%zu bytes, status=%d\n",
               perfResults[i].algo, perfResults[i].time,
               perfResults[i].memory, perfResults[i].status);
    }
}
```

## 实现版本对比

### 版本1: 基础实现 (Global Memory)
```cpp
__global__ void conv2dBasic(float *input, float *kernel, float *output,
                           int H, int W, int C, int Kh, int Kw, int F,
                           int stride, int padding) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int f = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (h < H && w < W && f < F) {
        float sum = 0.0f;
        
        for (int kh = 0; kh < Kh; kh++) {
            for (int kw = 0; kw < Kw; kw++) {
                for (int c = 0; c < C; c++) {
                    int ih = h * stride + kh - padding;
                    int iw = w * stride + kw - padding;
                    
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        sum += input[c * H * W + ih * W + iw] * 
                               kernel[f * C * Kh * Kw + c * Kh * Kw + kh * Kw + kw];
                    }
                }
            }
        }
        
        output[f * H * W + h * W + w] = sum;
    }
}
```

**问题分析:**
- 全局内存访问不合并
- 重复访问输入数据
- 内存带宽利用率低

### 版本2: 共享内存优化
```cpp
__global__ void conv2dShared(float *input, float *kernel, float *output,
                            int H, int W, int C, int Kh, int Kw, int F,
                            int stride, int padding) {
    __shared__ float s_input[TILE_H + 2*PADDING][TILE_W + 2*PADDING];
    __shared__ float s_kernel[KERNEL_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int h = by * TILE_H + ty;
    int w = bx * TILE_W + tx;
    
    // 协作加载输入数据到共享内存
    for (int c = 0; c < C; c++) {
        // 加载当前tile的数据
        if (ty < TILE_H && tx < TILE_W) {
            int ih = h * stride - padding;
            int iw = w * stride - padding;
            
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                s_input[ty + PADDING][tx + PADDING] = 
                    input[c * H * W + ih * W + iw];
            } else {
                s_input[ty + PADDING][tx + PADDING] = 0.0f;
            }
        }
        
        // 加载卷积核数据
        if (ty < Kh && tx < Kw) {
            s_kernel[ty * Kw + tx] = kernel[c * Kh * Kw + ty * Kw + tx];
        }
        
        __syncthreads();
        
        // 计算卷积
        if (h < H && w < W) {
            float sum = 0.0f;
            for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                    sum += s_input[ty + kh][tx + kw] * s_kernel[kh * Kw + kw];
                }
            }
            output[c * H * W + h * W + w] += sum;
        }
        
        __syncthreads();
    }
}
```

**优化点:**
- 使用共享内存减少全局内存访问
- 数据重用提高内存带宽利用率
- 协作加载提高内存合并访问

### 版本3: 分离卷积 (Separable Convolution)
```cpp
// 水平方向卷积
__global__ void conv2dHorizontal(float *input, float *kernel, float *output,
                                int H, int W, int C, int Kw) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < H && w < W) {
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int kw = 0; kw < Kw; kw++) {
                int iw = w + kw - Kw/2;
                if (iw >= 0 && iw < W) {
                    sum += input[c * H * W + h * W + iw] * kernel[kw];
                }
            }
            output[c * H * W + h * W + w] = sum;
        }
    }
}

// 垂直方向卷积
__global__ void conv2dVertical(float *input, float *kernel, float *output,
                              int H, int W, int C, int Kh) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < H && w < W) {
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int kh = 0; kh < Kh; kh++) {
                int ih = h + kh - Kh/2;
                if (ih >= 0 && ih < H) {
                    sum += input[c * H * W + ih * W + w] * kernel[kh];
                }
            }
            output[c * H * W + h * W + w] = sum;
        }
    }
}
```

**优势:**
- 将2D卷积分解为两个1D卷积
- 减少计算复杂度：O(Kh×Kw) → O(Kh + Kw)
- 适用于可分离的卷积核（如高斯核）

## 性能对比分析

### 1. 自定义实现 vs cuDNN
```cpp
void benchmarkConvolutionMethods(int H, int W, int C, int Kh, int Kw, int F,
                                int stride, int padding, int iterations) {
    // 分配内存
    size_t inputSize = H * W * C * sizeof(float);
    size_t kernelSize = F * C * Kh * Kw * sizeof(float);
    size_t outputSize = H * W * F * sizeof(float);
    
    float *h_input, *h_kernel, *h_output;
    float *d_input, *d_kernel, *d_output;
    
    cudaMallocHost(&h_input, inputSize);
    cudaMallocHost(&h_kernel, kernelSize);
    cudaMallocHost(&h_output, outputSize);
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaMalloc(&d_output, outputSize);
    
    // 初始化数据
    // ... 初始化代码 ...
    
    // 测试自定义实现
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        conv2dShared<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                           H, W, C, Kh, Kw, F, stride, padding);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto custom_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试cuDNN实现
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        conv2dCuDNN(h_input, h_kernel, h_output, 1, C, H, W, F, Kh, Kw,
                    padding, padding, stride, stride);
    }
    auto cudnn_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Custom Implementation: %ld μs\n", custom_time.count() / iterations);
    printf("cuDNN Implementation: %ld μs\n", cudnn_time.count() / iterations);
    printf("Speedup: %.2fx\n", (float)custom_time.count() / cudnn_time.count());
    
    // 清理资源
    cudaFreeHost(h_input);
    cudaFreeHost(h_kernel);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
```

### 2. 性能分析结果
- **小卷积核 (3×3)**: cuDNN通常快2-5倍
- **大卷积核 (7×7, 9×9)**: cuDNN通常快5-10倍
- **长序列**: cuDNN优势更明显
- **内存带宽**: cuDNN内存访问更优化

## 内存访问优化

### 1. 数据布局优化
```cpp
// 优化前：CHW格式
input[c * H * W + h * W + w]

// 优化后：HWC格式
input[h * W * C + w * C + c]
```

### 2. 内存合并访问
- 相邻线程访问相邻内存地址
- 使用向量化加载/存储指令
- 考虑内存对齐

### 3. 共享内存使用策略
- 合理选择tile大小
- 避免bank冲突
- 平衡共享内存使用和线程数

## 性能优化技巧

### 1. 线程块大小优化
```cpp
// 考虑因素
int threadsPerBlock = 256;  // 总线程数
int tileH = 16, tileW = 16;  // tile尺寸
int threadsPerTile = tileH * tileW;  // 每个tile的线程数
```

### 2. 循环展开
```cpp
// 手动展开循环
for (int kh = 0; kh < Kh; kh += 4) {
    sum += input[...] * kernel[kh] +
           input[...] * kernel[kh+1] +
           input[...] * kernel[kh+2] +
           input[...] * kernel[kh+3];
}
```

### 3. 使用纹理内存
```cpp
// 对于具有空间局部性的数据
texture<float, 2, cudaReadModeElementType> texInput;
// 在kernel中使用tex2D(texInput, x, y)
```

## CNN前向传播实现

### 1. 网络结构
```cpp
struct ConvLayer {
    int inputH, inputW, inputC;
    int outputH, outputW, outputC;
    int kernelH, kernelW;
    int stride, padding;
    float *weights, *bias;
};
```

### 2. 前向传播
```cpp
void forwardPass(float *input, float *output, ConvLayer *layer) {
    // 配置kernel参数
    dim3 blockDim(16, 16);
    dim3 gridDim((layer->outputW + 15) / 16, 
                  (layer->outputH + 15) / 16);
    
    // 启动卷积kernel
    conv2dKernel<<<gridDim, blockDim>>>(
        input, layer->weights, output,
        layer->inputH, layer->inputW, layer->inputC,
        layer->kernelH, layer->kernelW, layer->outputC,
        layer->stride, layer->padding
    );
    
    // 添加偏置项
    addBiasKernel<<<gridDim, blockDim>>>(
        output, layer->bias, layer->outputH, layer->outputW, layer->outputC
    );
}
```

## 编译和运行

### 编译命令
```bash
# 基础版本
nvcc -O3 -arch=sm_70 -o cnn_conv cnn_conv.cu

# 链接cuDNN
nvcc -O3 -arch=sm_70 -lcudnn -o cnn_conv_cudnn cnn_conv.cu

# 链接cuBLAS (用于矩阵乘法)
nvcc -O3 -arch=sm_70 -lcudnn -lcublas -o cnn_conv_full cnn_conv.cu
```

### 运行命令
```bash
./cnn_conv
```

## 性能基准测试

### 测试矩阵大小
- 输入尺寸: 224×224×3 (ImageNet标准)
- 卷积核: 3×3, 5×5, 7×7
- 输出通道: 64, 128, 256

### 性能指标
- FLOPS (每秒浮点运算次数)
- 内存带宽利用率
- 计算效率

## 常见问题和解决方案

### 1. 共享内存不足
- 减少tile大小
- 使用动态共享内存
- 重新设计算法

### 2. 边界处理
- 使用填充值
- 条件判断优化
- 考虑使用模板元编程

### 3. 精度问题
- 使用混合精度训练
- 考虑数值稳定性
- 验证计算结果

## 下一步
明天我们将学习注意力机制(Attention)和Transformer的实现，这是现代NLP的基础。

## 练习
1. 实现不同卷积核大小的版本，比较性能
2. 添加批处理支持，处理多个输入
3. 实现卷积层的反向传播
4. 使用cuDNN库对比性能
5. 实现Winograd卷积算法优化

## 参考资料
- [CUDA Convolution Implementation](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- [cuDNN Library](https://docs.nvidia.com/deeplearning/cudnn/)
- [CNN Architecture Design](https://arxiv.org/abs/1512.03385)
- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [cuDNN Performance Guide](https://docs.nvidia.com/deeplearning/cudnn/performance-guide/)
- [Convolution Algorithms](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#cudnnConvolutionFwdAlgo_t)
- [Winograd Convolution](https://arxiv.org/abs/1509.09308)
- [Fast Algorithms for Convolutional Neural Networks](https://arxiv.org/abs/1509.09308)
- [cuDNN Convolution Performance](https://developer.nvidia.com/cudnn)
- [GPU Memory Management](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [Shared Memory Optimization](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [Convolution Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [ImageNet Dataset](https://image-net.org/)
- [ResNet Architecture](https://arxiv.org/abs/1512.03385)
- [VGG Architecture](https://arxiv.org/abs/1409.1556)
- [AlexNet Architecture](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

---

## 📁 相关文件快速链接
本教程包含以下相关程序文件，点击即可查看：

### 🚀 示例程序
- [`cnn_conv.cu`](cnn_conv.cu) - 基础2D卷积实现
- [`cnn_conv_optimized.cu`](cnn_conv_optimized.cu) - 优化版本卷积
- [`cnn_conv_cudnn.cu`](cnn_conv_cudnn.cu) - cuDNN版本卷积
- [`cnn_forward.cu`](cnn_forward.cu) - CNN前向传播实现

### 📊 性能分析工具
- 使用`nvprof`进行命令行性能分析
- 使用Nsight Systems进行系统级性能分析
- 使用Nsight Compute进行kernel级性能分析

### 🔧 优化技巧
- 共享内存tile优化
- 内存合并访问优化
- Winograd卷积算法
- 混合精度计算
