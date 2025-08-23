# Day 1: CUDA编程基础 - 向量加法

## 概述
今天我们将开始CUDA编程之旅，从最基础的向量加法开始。这是理解CUDA编程模型和GPU并行计算的第一步。

## 学习目标
- 理解CUDA编程模型的基本概念
- 学会编写第一个CUDA kernel
- 掌握CUDA内存管理的基本操作
- 理解线程块(Block)和网格(Grid)的概念

## 开发环境说明
本教程主要基于**NVIDIA GPU**进行讲解，笔者的开发环境使用：
- **操作系统**: Linux Ubuntu 22.04 LTS
- **GPU**: NVIDIA GeForce RTX 4090 (Ada Lovelace架构)
- **CUDA版本**: CUDA 12.4
- **编译器**: nvcc (NVIDIA CUDA Compiler)

虽然不同GPU型号在具体参数上有所差异，但CUDA编程的基本概念和API是一致的。

## GPU硬件基础

### GPU vs CPU架构差异

#### CPU架构特点
- **串行执行**: 少量强大的核心，适合复杂的串行任务
- **大缓存**: 多级缓存系统，减少内存访问延迟
- **分支预测**: 智能的分支预测和乱序执行
- **通用性**: 适合各种类型的计算任务

#### GPU架构特点
- **并行执行**: 大量简单的核心，专为并行计算设计
- **内存层次**: 全局内存、共享内存、寄存器等多层内存
- **SIMT模型**: Single Instruction, Multiple Thread执行模型
- **专用性**: 特别适合数据并行的计算密集型任务

### GPU硬件架构和内存布局

#### GPU整体架构
```
┌───────────────────────────────────────────────────────────────────┐
│                          GPU Device                               │
├───────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐     ┌───────────────┐       │
│  │     SM 0      │  │     SM 1      │     │     SM N      │       │
│  │  (Streaming   │  │  (Streaming   │ ... │  (Streaming   │       │
│  │Multiprocessor)│  │Multiprocessor)│     │Multiprocessor)│       │
│  └───────────────┘  └───────────────┘     └───────────────┘       │
├───────────────────────────────────────────────────────────────────┤
│                           L2 Cache                                │
├───────────────────────────────────────────────────────────────────┤
│                     Global Memory (HBM/GDDR)                      │
└───────────────────────────────────────────────────────────────────┘
```

#### 单个SM内部结构 (ASCII图)
```
┌───────────────────────────────────────────────────────────────────┐
│                           Single SM                               │
├───────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│  │    Warp 0     │  │    Warp 1     │  │    Warp N     │          │
│  │  (32 threads) │  │  (32 threads) │  │  (32 threads) │          │
│  └───────────────┘  └───────────────┘  └───────────────┘          │
├───────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│  │    Register   │  │    Shared     │  │   L1 Cache    │          │
│  │     File      │  │    Memory     │  │               │          │
│  │    (64KB)     │  │    (48KB)     │  │    (32KB)     │          │
│  └───────────────┘  └───────────────┘  └───────────────┘          │
├───────────────────────────────────────────────────────────────────┤
│                    Special Function Units                         │
│              (FP32, FP64, INT, Tensor Cores)                      │
└───────────────────────────────────────────────────────────────────┘
```

#### GPU内存层次结构 (ASCII图)
```
┌───────────────────────────────────────────────────────────────────┐
│                        Memory Hierarchy                           │
├───────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│  │  Registers    │  │  Shared       │  │  Local        │          │
│  │  (Fastest)    │  │  Memory       │  │  Memory       │          │
│  │  255/thread   │  │  48KB/block   │  │  Global       │          │
│  │               │  │               │  │  based        │          │
│  └───────────────┘  └───────────────┘  └───────────────┘          │
├───────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│  │  Constant     │  │  Texture      │  │  Global       │          │
│  │  Memory       │  │  Memory       │  │  Memory       │          │
│  │  (Cached)     │  │  (Optimized)  │  │  (Slowest)    │          │
│  │               │  │               │  │               │          │
│  └───────────────┘  └───────────────┘  └───────────────┘          │
└───────────────────────────────────────────────────────────────────┘
```

### GPU内存层次详解

#### 1. 寄存器 (Registers)
- **特点**: 最快的访问速度，每个线程私有
- **容量**: 每个线程最多255个32位寄存器
- **用途**: 存储频繁访问的变量和中间结果

#### 2. 共享内存 (Shared Memory)
- **特点**: 线程块内共享，访问延迟低
- **容量**: 每个线程块最多48KB (计算能力8.x)
- **用途**: 线程间数据交换和协作

#### 3. 本地内存 (Local Memory)
- **特点**: 每个线程私有，但存储在全局内存中
- **用途**: 存储大型数组和寄存器溢出数据

#### 4. 全局内存 (Global Memory)
- **特点**: 所有线程可访问，容量大但延迟高
- **用途**: 存储输入数据和计算结果

#### 5. 常量内存 (Constant Memory)
- **特点**: 只读，有缓存机制
- **用途**: 存储kernel参数和查找表

#### 6. 纹理内存 (Texture Memory)
- **特点**: 优化的2D/3D数据访问
- **用途**: 图像处理和科学计算

## CUDA编程模型基础

### 1. 主机代码 vs 设备代码
- **主机代码(Host Code)**: 在CPU上运行，负责数据准备、内存分配和结果处理
- **设备代码(Device Code)**: 在GPU上运行，使用`__global__`关键字标记

### 2. 线程层次结构
```
Grid (网格)
├── Block 0 (线程块0)
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
├── Block 1 (线程块1)
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
└── ...
```

### 3. 线程索引计算
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
- `blockIdx.x`: 当前线程块在网格中的索引
- `blockDim.x`: 每个线程块中的线程数
- `threadIdx.x`: 当前线程在线程块中的索引

## 代码解析

### Kernel函数
```cpp
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**关键点:**
- `__global__`: 表示这是一个CUDA kernel，从主机调用，在设备上执行
- 每个线程处理一个数组元素
- 边界检查确保不会越界访问

### 主机代码流程
1. **内存分配**: 在主机和设备上分配内存
2. **数据传输**: 将数据从主机复制到设备
3. **Kernel启动**: 配置线程块和网格大小，启动kernel
4. **结果获取**: 将结果从设备复制回主机

### 线程配置
```cpp
int threadsPerBlock = 256;  // 每个线程块256个线程
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;  // 计算需要的线程块数
vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
```

## 编译和运行

### 编译命令
```bash
nvcc -o vector_add vector_add.cu
```

### 运行命令
```bash
./vector_add
```

## CUDA执行模型

#### Warp概念
- **定义**: 32个线程组成一个warp，是GPU调度的基本单位
- **执行**: 同一个warp内的线程执行相同的指令
- **分支**: 如果warp内线程执行不同分支，会导致warp分化(warp divergence)

#### 线程块调度
- **SM分配**: 线程块被分配到不同的流式多处理器(SM)
- **资源限制**: 每个SM有固定的寄存器、共享内存和线程块数量
- **动态调度**: GPU自动管理线程块的调度和执行

#### 内存合并访问
- **概念**: 多个线程同时访问连续的内存地址
- **优化**: 合并访问可以提高内存带宽利用率
- **实现**: 使用合适的线程索引计算模式

### 性能优化基础

#### 内存访问模式
```cpp
// 好的访问模式 - 合并访问
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float value = data[idx];  // 连续访问

// 避免的访问模式 - 分散访问
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float value = data[idx * stride];  // 可能不连续
```

#### 共享内存使用
```cpp
__shared__ float shared_data[BLOCK_SIZE];

// 协作加载数据
int tid = threadIdx.x;
shared_data[tid] = global_data[blockIdx.x * blockDim.x + tid];
__syncthreads();  // 确保所有线程都加载完成

// 使用共享内存进行计算
float result = shared_data[tid] + shared_data[tid + 1];
```

#### 寄存器优化
```cpp
// 避免寄存器溢出
__global__ void optimized_kernel(float *data) {
    // 使用寄存器存储频繁访问的值
    float local_sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        local_sum += data[i];
    }
    // 最后一次性写入全局内存
}
```

## 性能分析

### 为什么选择256线程/块？
- GPU的warp大小是32，256是32的倍数
- 平衡了寄存器使用和线程切换开销
- 适合大多数GPU架构

### 内存带宽
- 每个线程读取2个float，写入1个float
- 理论内存带宽利用率取决于GPU架构

### 性能指标计算

#### 理论峰值性能
```cpp
// RTX 4090的理论峰值
float peak_gflops = 16384 * 2 * 2.52 / 1000;  // 约165 TFLOPS (FP32)
float peak_memory_bandwidth = 1008.0;  // GB/s
```

#### 实际性能测量
```cpp
// 使用CUDA事件测量时间
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

// 计算性能
float gflops = (n * 2) / (milliseconds * 1e6);  // 2个浮点运算
float memory_bandwidth = (n * 3 * sizeof(float)) / (milliseconds * 1e6);  // GB/s
```

### 性能优化策略

#### 1. 内存访问优化
- **合并访问**: 确保线程访问连续内存地址
- **内存对齐**: 使用适当的内存对齐
- **缓存友好**: 利用GPU的L2缓存

#### 2. 计算优化
- **循环展开**: 减少循环开销
- **指令级并行**: 利用GPU的指令流水线
- **数学函数**: 使用快速数学函数(`__expf`, `__logf`等)

#### 3. 资源利用
- **占用率**: 保持足够的线程块在SM上
- **寄存器使用**: 平衡寄存器数量和线程数量
- **共享内存**: 合理使用共享内存减少全局内存访问

### 性能分析工具

#### NVIDIA Visual Profiler
```bash
# 使用nvprof进行性能分析
nvprof ./vector_add

# 详细分析
nvprof --metrics all ./vector_add
```

#### Nsight Systems
- 系统级性能分析
- 显示CPU和GPU的协作情况
- 识别瓶颈和优化机会

#### Nsight Compute
- 详细的kernel性能分析
- 寄存器使用、共享内存使用等指标
- 提供具体的优化建议

## 常见问题和调试技巧

### 1. 内存错误
- 使用`cuda-memcheck`工具检查内存访问错误
- 确保所有内存分配都成功

### 2. Kernel启动失败
- 检查线程块配置是否超出硬件限制
- 使用`cudaGetLastError()`检查错误

### 3. 性能优化
- 使用`nvprof`或Nsight Systems分析性能
- 监控内存传输和kernel执行时间

## 下一步
明天我们将学习如何手动加载PTX代码，深入了解CUDA的底层机制。

## 练习
1. 修改代码实现向量减法
2. 尝试不同的线程块大小，观察性能变化
3. 添加错误检查代码，提高程序的健壮性

## 参考资料
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [NVIDIA GPU Architecture](https://www.nvidia.com/en-us/data-center/gpu-architecture/)
- [CUDA Performance Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU Computing Gems](https://developer.nvidia.com/gpugems)
- [NVIDIA Technical Blog](https://developer.nvidia.com/blog/)

### 学术论文
- **"CUDA: Scalable Parallel Programming for GPUs"** - Nickolls et al., 2008
- **"Understanding the Efficiency of GPU Algorithms for Matrix-Matrix Multiplication"** - Volkov & Demmel, 2008
- **"Optimizing CUDA Code for Fermi Architecture"** - NVIDIA, 2010
- **"GPU Memory Model and Programming Model"** - NVIDIA, 2012
- **"Maxwell: The Most Advanced CUDA GPU Ever Made"** - NVIDIA, 2014
- **"Volta: The Most Advanced Data Center GPU"** - NVIDIA, 2017
- **"Turing: The Most Advanced GPU Architecture"** - NVIDIA, 2018
- **"Ampere: NVIDIA's Next Generation Data Center GPU"** - NVIDIA, 2020
- **"Hopper: NVIDIA's Next Generation GPU Architecture"** - NVIDIA, 2022
- **"Ada Lovelace: NVIDIA's Next Generation Gaming GPU"** - NVIDIA, 2022
- **"Blackwell: NVIDIA's Next Generation AI GPU"** - NVIDIA, 2024

### 在线资源
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/deep-learning-ai/education-training/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

---

## 附录：GPU硬件详细信息

### NVIDIA GPU架构演进历史

#### 数据中心GPU架构
- **[Tesla (2006)](https://www.nvidia.com/en-us/data-center/tesla/)**: 第一代统一着色器架构，引入CUDA编程模型
- **[Fermi (2010)](https://www.nvidia.com/en-us/data-center/tesla/)**: 引入真正的缓存层次和共享内存，支持ECC内存
- **[Kepler (2012)](https://www.nvidia.com/en-us/data-center/tesla/)**: 动态并行和Hyper-Q技术，显著提升并行处理能力
- **[Maxwell (2014)](https://www.nvidia.com/en-us/data-center/tesla/)**: 能效比显著提升，引入动态并行
- **[Pascal (2016)](https://www.nvidia.com/en-us/data-center/tesla/)**: 引入NVLink和统一内存，支持HBM2显存
- **[Volta (2017)](https://www.nvidia.com/en-us/data-center/tesla/)**: Tensor Core和独立线程调度，专为AI训练设计
- **[Turing (2018)](https://www.nvidia.com/en-us/data-center/tesla/)**: RT Core和AI加速，支持实时光线追踪
- **[Ampere (2020)](https://www.nvidia.com/en-us/data-center/tesla/)**: 第三代Tensor Core和RT Core，支持稀疏计算
- **[Hopper (2022)](https://www.nvidia.com/en-us/data-center/tesla/)**: 第四代Tensor Core和Transformer Engine，专为AI推理优化
- **[Blackwell (2024)](https://www.nvidia.com/en-us/data-center/tesla/)**: 第五代Tensor Core和AI推理加速，支持万亿参数模型

#### 消费级GPU架构
- **[Maxwell (2014)](https://www.nvidia.com/en-us/geforce/)**: GTX 900系列，能效比革命性提升
- **[Pascal (2016)](https://www.nvidia.com/en-us/geforce/)**: GTX 1000系列，引入VRWorks和Ansel
- **[Turing (2018)](https://www.nvidia.com/en-us/geforce/)**: RTX 2000系列，实时光线追踪和DLSS
- **[Ampere (2020)](https://www.nvidia.com/en-us/geforce/)**: RTX 3000系列，第二代RT Core和第三代Tensor Core
- **[Ada Lovelace (2022)](https://www.nvidia.com/en-us/geforce/)**: RTX 4000系列，第四代RT Core和DLSS 3

### RTX 4090详细规格 (Ada Lovelace架构)

#### 核心规格
- **CUDA核心**: 16,384个
- **Tensor核心**: 512个 (第四代)
- **RT核心**: 144个 (第四代)
- **计算能力**: 8.9
- **基础频率**: 2.23 GHz
- **Boost频率**: 2.52 GHz

#### 内存规格
- **显存容量**: 24GB GDDR6X
- **显存带宽**: 1,008 GB/s
- **显存位宽**: 384-bit
- **显存频率**: 21 Gbps

#### 功耗和散热
- **最大功耗**: 450W
- **推荐电源**: 850W
- **散热设计**: 三风扇，双槽位
- **接口**: PCIe 4.0 x16

#### 官方链接
- [RTX 4090产品页面](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
- [RTX 4090技术规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
- [Ada Lovelace架构白皮书](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/)

### GPU内存层次详细规格

#### 寄存器 (Registers)
- **访问延迟**: 1个时钟周期
- **容量**: 每个线程最多255个32位寄存器
- **带宽**: 理论无限 (受限于硬件)
- **用途**: 存储频繁访问的变量和中间结果
- **优化建议**: 避免寄存器溢出，合理使用循环展开

#### 共享内存 (Shared Memory)
- **访问延迟**: 1-2个时钟周期
- **容量**: 每个线程块最多48KB (计算能力8.x)
- **带宽**: 约1.6 TB/s
- **用途**: 线程间数据交换和协作
- **优化建议**: 避免bank冲突，合理分配内存

#### 本地内存 (Local Memory)
- **访问延迟**: 200-800个时钟周期
- **容量**: 受全局内存限制
- **带宽**: 受全局内存带宽限制
- **用途**: 存储大型数组和寄存器溢出数据
- **优化建议**: 尽量避免使用，优先使用寄存器

#### 全局内存 (Global Memory)
- **访问延迟**: 200-800个时钟周期
- **容量**: 受GPU显存限制
- **带宽**: RTX 4090约1,008 GB/s
- **用途**: 存储输入数据和计算结果
- **优化建议**: 使用合并访问，合理内存对齐

#### 常量内存 (Constant Memory)
- **访问延迟**: 1-2个时钟周期 (缓存命中)
- **容量**: 64KB (每个SM)
- **带宽**: 约1.6 TB/s
- **用途**: 存储kernel参数和查找表
- **优化建议**: 适合广播访问模式

#### 纹理内存 (Texture Memory)
- **访问延迟**: 1-2个时钟周期 (缓存命中)
- **容量**: 受全局内存限制
- **带宽**: 约1.6 TB/s
- **用途**: 图像处理和科学计算
- **优化建议**: 适合2D/3D空间局部性访问

### 性能基准和对比

#### 理论峰值性能
- **FP32性能**: 约165 TFLOPS
- **FP16性能**: 约330 TFLOPS (使用Tensor Core)
- **内存带宽**: 1,008 GB/s
- **功耗效率**: 约0.37 TFLOPS/W

#### 实际应用性能
- **深度学习训练**: 比RTX 3090快约1.5-2倍
- **科学计算**: 比RTX 3090快约1.3-1.8倍
- **游戏性能**: 4K分辨率下比RTX 3090快约1.2-1.5倍
- **光线追踪**: 比RTX 3090快约1.4-2倍

### 相关技术文档链接

#### NVIDIA官方文档
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA性能最佳实践](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU架构白皮书](https://www.nvidia.com/en-us/data-center/gpu-architecture/)
- [开发者资源中心](https://developer.nvidia.com/)

#### 技术博客和论坛
- [NVIDIA开发者博客](https://developer.nvidia.com/blog/)
- [CUDA开发者论坛](https://forums.developer.nvidia.com/)
- [GPU计算社区](https://developer.nvidia.com/gpu-computing)
- [深度学习学院](https://www.nvidia.com/en-us/deep-learning-ai/education-training/)
