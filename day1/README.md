# Day 1: CUDA编程基础 - 硬件架构与编程模型

## 概述
今天我们将开始CUDA编程之旅，从GPU硬件架构和CUDA编程模型的基础概念开始。通过理解GPU的硬件特性、内存层次结构和线程执行模型，为后续的CUDA编程打下坚实基础。我们将通过向量加法和向量点积的示例来实践这些概念。

## 学习目标
- 理解CUDA编程模型的基本概念（Host vs Device, Kernel, Grid, Block, Thread）
- 掌握GPU硬件架构和内存层次结构（SM, Warp, 寄存器, 共享内存等）
- 学会编写简单的CUDA kernel函数（向量加法、向量点积）
- 理解线程层次结构和索引计算（Grid-Block-Thread关系）
- 掌握CUDA内存管理基础（cudaMalloc, cudaMemcpy等）
- 学会使用GPU配置工具获取硬件参数和优化建议
- 理解动态kernel加载机制（CUBIN文件, Driver API）
- 掌握Warp执行特性和避免Warp分化的基本方法

## 开发环境说明
本教程主要基于**NVIDIA GPU**进行讲解，笔者的开发环境使用：
- **操作系统**: Linux Ubuntu 22.04 LTS
- **GPU**: NVIDIA GeForce RTX 4090 (Ada Lovelace架构)
- **CUDA版本**: CUDA 12.4
- **编译器**: nvcc (NVIDIA CUDA Compiler)

> 📋 **环境搭建**: 如果你还没有安装CUDA开发环境(包括nvcc)，请参考 [环境搭建指南](env.md) 进行安装和配置。

虽然不同GPU型号在具体参数上有所差异，但CUDA编程的基本概念和API是一致的。



## GPU硬件基础

### GPU vs CPU架构差异

#### CPU架构特点
CPU（中央处理器）采用串行执行架构，配备少量但功能强大的计算核心。每个核心都具备复杂的控制逻辑和分支预测能力，能够智能地预测程序执行路径并实现乱序执行，从而最大化指令级并行性。CPU拥有庞大的多级缓存系统，包括L1、L2、L3缓存，这些缓存层次能够有效减少内存访问延迟，提高数据局部性。由于CPU的设计目标是通用性，它能够高效地处理各种类型的计算任务，从复杂的控制逻辑到简单的算术运算都能胜任。

#### GPU架构特点
GPU（图形处理器）则采用并行执行架构，配备大量相对简单的计算核心。这些核心虽然单个能力不如CPU核心强大，但数量众多，能够同时处理大量相似的并行任务。GPU采用SIMT（Single Instruction, Multiple Thread）执行模型，即多个线程执行相同的指令但处理不同的数据，这种设计特别适合数据并行的计算密集型任务。在内存系统方面，GPU采用层次化的内存设计，包括全局内存、共享内存、寄存器等多个层次，每个层次都有不同的访问延迟和容量特性。由于GPU的设计目标明确，它在图像处理、科学计算、深度学习等并行计算领域表现出色，但在复杂的串行任务上不如CPU高效。

### GPU硬件架构和内存布局

#### GPU整体架构示意图
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

现代GPU采用分层架构设计，从宏观到微观可以分为三个主要层次。最上层是GPU设备整体，包含多个流式多处理器（Streaming Multiprocessor, SM），这些SM是GPU的核心计算单元，每个SM都具备独立的指令调度和执行能力。在SM层之下是共享的L2缓存，为所有SM提供统一的数据缓存服务。最底层是全局内存，通常采用HBM（High Bandwidth Memory）或GDDR技术，为整个GPU提供大容量、高带宽的存储空间。

#### 单个SM内部结构示意图
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

每个SM内部采用细粒度的并行设计。在SM的顶层，多个Warp并行执行，这些Warp共享SM的计算资源。

> **Warp概念详解**：
Warp是GPU调度的基本单位，每个Warp包含32个线程。Warp采用SIMT（Single Instruction, Multiple Thread）执行模型，即同一Warp内的所有线程执行相同的指令，但处理不同的数据。这种设计使得GPU能够高效地利用数据并行性，当多个线程需要执行相同操作时，只需要一条指令就能控制32个线程同时执行。**🎯注意**: Warp大小32是NVIDIA GPU架构的固定设计，从最早的Tesla架构到最新的Blackwell架构都保持不变。32这个数字是经过精心设计以便能够充分利用GPU的SIMT执行单元.

在Warp层之下，SM配备了三种关键的内存资源：寄存器文件（Register File）、共享内存（Shared Memory）和L1缓存。寄存器文件为每个线程提供最快的存储访问，共享内存支持线程块内的数据交换和协作，L1缓存则提供额外的数据缓存层。在SM的最底层，配置了各种专用功能单元，包括FP32、FP64浮点运算单元、整数运算单元以及Tensor Core等专用加速器。


#### GPU内存层次结构示意图
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

GPU的内存系统采用层次化设计，从访问速度最快到最慢依次为：寄存器、共享内存、本地内存、常量内存、纹理内存和全局内存。寄存器是每个线程私有的存储空间，访问延迟仅需1个时钟周期，但容量有限，每个线程最多只能使用255个32位寄存器。共享内存是线程块内共享的存储空间，访问延迟为1-2个时钟周期，容量为每个线程块48KB，适合存储频繁访问的中间结果和实现线程间协作。本地内存虽然名义上是线程私有，但实际存储在全局内存中，访问延迟较高（200-800个时钟周期），主要用于存储大型数组和寄存器溢出的数据。常量内存具有缓存机制，适合存储kernel参数和查找表，当多个线程访问相同地址时性能最佳。纹理内存针对2D/3D空间局部性访问进行了优化，适合图像处理和科学计算应用。全局内存容量最大但访问延迟最高，是所有线程共享的主要存储空间，其性能很大程度上依赖于内存访问模式，合并访问（Coalesced Access）可以显著提高内存带宽利用率。

## CUDA编程模型基础

学习CUDA编程，最好的方式就是通过一个具体的例子来理解。让我们先看看一个完整的向量加法程序，感受一下CUDA代码是什么样子的：

```cpp
// 向量加法的CUDA kernel函数
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
// 主机代码调用GPU kernel
int main() {
    // ... 内存分配和数据准备 ...
    // 启动kernel，配置线程网格
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // n = 1M
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    // ... 结果获取和清理 ...
}
```

看到这段代码，你可能会有很多疑问：`__global__`是什么？`<<<>>>`这个奇怪的语法又是什么意思？为什么需要计算`blocksPerGrid`？这些正是我们要学习的核心概念。

我们选择向量加法作为入门示例，正是因为它足够简单，能够让我们专注于理解CUDA的编程模型，而不是被复杂的算法逻辑分散注意力。通过这个例子，我们将一步步地走进CUDA的世界，理解它的设计哲学和运行机制。

## 理解CUDA的双重世界

回到我们的向量加法程序，你可能会注意到代码被分成了两个部分：一个是在CPU上运行的主函数，另一个是在GPU上执行的kernel函数。这就是CUDA编程的核心思想——**分离关注点**。

### 主机代码：CPU的指挥中心

在CUDA编程中，CPU扮演着"指挥中心"的角色。它负责整个程序的流程控制，就像乐队的指挥家一样，不直接演奏乐器，但协调着整个演出的节奏和顺序。

CPU上的代码（我们称为主机代码）使用标准的C++编写，它的主要工作包括：
- 准备数据，就像厨师准备食材一样
- 在CPU内存中分配空间，为数据准备"临时存放点"
- 在GPU内存中分配空间，就像为GPU准备"工作台"
- 启动GPU kernel，相当于给GPU下达"开始工作"的指令
- 从GPU获取计算结果，就像收集完成的作品
- 处理结果并输出，进行最后的整理和展示

### 设备代码：GPU的并行工厂

而GPU上的代码（设备代码）则专注于计算本身。它使用CUDA C++的扩展语法，通过`__global__`这样的关键字告诉编译器："这段代码要在GPU上运行"。

GPU就像一个巨大的并行工厂，每个线程都是一个小工人，同时处理不同的数据。在我们的向量加法中，每个线程负责计算一个元素的和，这样就能同时处理成千上万个元素，而不是像CPU那样一个一个地计算。

## 线程的组织艺术

现在让我们深入理解GPU是如何组织这些"小工人"的。想象一下，如果你要管理一个大型工厂，你会怎么组织工人呢？

### 从工厂管理看线程层次

GPU采用了三层管理结构，就像现代工厂的管理体系：

**第一层：工厂整体（Grid）**
整个GPU就像一个大型工厂，我们称之为Grid（网格）。这个工厂可以同时运行多个生产车间，每个车间负责不同的任务。在我们的向量加法中，Grid就是所有需要计算的元素的总集合。

**第二层：生产车间（Block）**
每个生产车间（Block）都有固定数量的工人，这些工人可以相互协作，共享工具和材料。Block内的线程可以访问同一块共享内存，就像车间里的工人可以共用工具箱一样。

**第三层：工人（Thread）**
每个工人（Thread）负责具体的计算任务。在我们的例子中，每个线程计算一个向量元素的和。虽然所有线程执行相同的代码，但处理的是不同的数据。

```
工厂 (Grid)
├── 车间0 (Block 0)
│   ├── 工人0 (Thread 0)
│   ├── 工人1 (Thread 1)
│   └── ...
├── 车间1 (Block 1)
│   ├── 工人0 (Thread 0)
│   ├── 工人1 (Thread 1)
│   └── ...
└── ...
```

### 给每个工人分配任务

现在的问题是：如何让每个工人知道自己应该处理哪个数据？这就是线程索引计算的核心。

在我们的向量加法中，每个线程需要计算：
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

这个公式就像给工厂里的每个工人分配工号一样。让我用一个更贴近生活的例子来解释：

### 学校点名系统的启发

想象一下学校里的点名系统。假设学校有3个班级，每个班级有4个学生，要给全校学生编号，你会怎么设计？

**班级结构**：
- 班级0：学生0(编号0), 学生1(编号1), 学生2(编号2), 学生3(编号3)
- 班级1：学生0(编号4), 学生1(编号5), 学生2(编号6), 学生3(编号7)  
- 班级2：学生0(编号8), 学生1(编号9), 学生2(编号10), 学生3(编号11)

**编号计算公式**：
```
学生编号 = 班级号 × 每班人数 + 班级内学号
```

例如：
- 班级1的学生2：1 × 4 + 2 = 6号
- 班级2的学生1：2 × 4 + 1 = 9号

**CUDA中的对应关系**：
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
- `blockIdx.x` = 班级号（0, 1, 2...）
- `blockDim.x` = 每班人数（4）
- `threadIdx.x` = 班级内学号（0, 1, 2, 3）
- `idx` = 全局学生编号

这样，每个线程就能知道自己应该处理数组中的哪个元素了。通过这种巧妙的索引设计，GPU能够高效地分配工作，让每个线程都有明确的任务，从而实现真正的并行计算。

> 🎯注意: 这里的idx计算只显式用了x维度，没有用到y和z维度，因为Grid和Block都是1维的.这个时候1维写法kernel<<< numBlocks, threadsPerBlock >>>(参数...)等价于3维写法（y和z都是1）kernel<<< dim3(numBlocks, 1, 1), dim3(threadsPerBlock, 1, 1) >>>(参数...). 详细情况参考[CUDA编程模型深入：从抽象到现实](gbwt.md).

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

**完整代码**: [`vector_add.cu`](vector_add.cu)

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

#### 基本配置
```cpp
int threadsPerBlock = 256;  // 每个线程块256个线程
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;  // 计算需要的线程块数 n = 1M
vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
```

#### 参数说明和优化

**为什么选择256个线程？**
256这个数字是经过大量实践测试得出的经验值，它平衡了以下几个因素：
- **硬件限制**: 每个SM最多支持的线程数因GPU架构而异，RTX 4090支持1024个线程/SM，256是一个常用的平衡值
- **资源利用率**: 256个线程能够充分利用SM的寄存器、共享内存等资源
- **调度效率**: 是32的倍数（warp大小），避免warp分化
- **灵活性**: 256 = 8 × 32，可以灵活地组织成2D或3D的线程块结构

**为什么是32的倍数？**
由于warp大小固定为32，选择32的倍数作为线程块大小可以：
- 确保每个warp都能被完全填满，避免部分warp的浪费
- 优化GPU的调度效率，减少线程切换开销
- 提高内存访问的合并性，增加内存带宽利用率


**✍️什么是Warp分化(Divergence)？**
Warp分化是GPU编程中的一个重要概念，指的是同一个warp内的线程执行不同的代码路径。当warp内的32个线程遇到条件分支时，GPU无法让所有线程同时执行，必须串行处理每个分支，这会导致严重的性能损失。

```cpp
// 示例：会导致warp分化的代码
__global__ void divergent_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // 条件分支 - 可能导致warp分化
        if (data[idx] > 0) {  // 另一个条件分支
            data[idx] = data[idx] * 2;  // 部分线程执行
        } else {
            data[idx] = data[idx] / 2;  // 其他线程执行
        }
    }
}
```

在上面的代码中，当不同的线程执行不同的条件分支时，就会发生warp分化。GPU需要先执行所有满足`data[idx] > 0`条件的线程，然后再执行其他线程，这种串行执行方式大大降低了并行效率。


**💡 获取GPU参数的小程序**
为了方便获取这些参数值，我们提供了专门的GPU信息获取程序：

- **Python版本**: [`gpu_info.py`](gpu_info.py)
- **CUDA版本**: [`gpu_info.cu`](gpu_info.cu)
- **C++版本**: [`gpu_info_cpp.cpp`](gpu_info_cpp.cpp)
- **配置总结**: [`GPU_CONFIG_SUMMARY.md`](GPU_CONFIG_SUMMARY.md) - 笔者的RTX 4090详细配置

运行这些程序可以获取：
- `prop.maxThreadsPerBlock` - 每块最大线程数
- `prop.maxThreadsPerMultiProcessor` - 每SM最大线程数
- `prop.sharedMemPerBlock` - 每块最大共享内存
- `prop.regsPerBlock` - 每块最大寄存器数
- `prop.multiProcessorCount` - SM数量
- 以及更多GPU硬件参数

**推荐使用方式**：
```bash
# Python版本（推荐，无需编译）
python gpu_info.py

# C++版本（系统信息检查）
g++ -o gpu_info_cpp gpu_info_cpp.cpp
./gpu_info_cpp

# CUDA版本（完整硬件信息，需要编译）
nvcc -O3 -o gpu_info gpu_info.cu
./gpu_info
```

**blocksPerGrid计算公式解释**：
```cpp
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerGrid; // n = 1M
```
这个公式确保有足够的线程块来处理所有数据：
- 如果n=1000，threadsPerBlock=256
- 需要4个线程块：前3个处理256×3=768个元素，第4个处理剩余的232个元素
- 公式中的`+ threadsPerBlock - 1`是为了向上取整

**实际应用建议**：
- **小数据集**（n < 10000）: 使用64-128个线程
- **中等数据集**（10000 ≤ n < 1000000）: 使用256个线程
- **大数据集**（n ≥ 1000000）: 使用512个线程
- **特殊应用**: 根据具体算法特点调整，如矩阵运算可能需要2D线程块

## 编译和运行

### 编译命令
```bash
nvcc -o vector_add vector_add.cu
```

**源代码**: [`vector_add.cu`](vector_add.cu)

### 运行命令并观察结果
```bash
$./vector_add 
c[0] = 3
c[1] = 3
c[2] = 3
```

### 🚀 动态Kernel加载系统

#### CUBIN文件

在深入动态kernel加载之前，我们需要先了解CUBIN文件。CUBIN（CUDA Binary）是CUDA编译器`nvcc`生成的二进制文件，包含了编译后的GPU机器码。CUBIN是**特定GPU架构的二进制格式**，可以直接被GPU执行。

**CUBIN文件的生成过程：**
```bash
# 从CUDA源码(.cu)生成PTX中间文件
nvcc -ptx -o vector_add.ptx vector_add.cu

# 从PTX生成CUBIN二进制文件  
nvcc -cubin -o vector_add.cubin vector_add.ptx

# 或者直接从源码一步生成CUBIN
nvcc -cubin -o vector_add.cubin vector_add.cu
```

#### 什么是动态Kernel加载？
动态kernel加载是指程序可以在运行时动态选择要执行的kernel函数，按需加载不同的GPU Kernel(CUBIN文件)，灵活配置kernel的执行参数，最终实现插件化的GPU计算系统。程序可以根据运行时条件选择不同的算法，无需重新编译就能添加新功能，实现真正的可扩展性。同时，这种技术能够根据硬件和数据特征选择最优kernel，提升性能表现。更重要的是，它将不同的计算功能分离到独立的CUBIN文件中，实现了良好的模块化设计。

#### 实例演示

我们通过两个不同的向量运算kernel来演示动态加载。首先需要生成多个CUBIN文件，每个文件包含不同的计算功能：

```bash
# 1. 生成向量加法kernel的CUBIN文件
nvcc -cubin -arch=sm_89 -o vector_add.cubin vector_add.cu
# 2. 生成向量点积kernel的CUBIN文件  
nvcc -cubin -o vector_dot.cubin vector_dot.cu
# 3. 编译动态加载程序
nvcc run_cubin.cpp -lcuda -o run_cubin
# 4. 运行时选择不同的kernel
./run_cubin vector_add.cubin    # 执行向量加法: [1,2,3] + [2,2,2] = [3,4,5]
./run_cubin vector_dot.cubin    # 执行向量点积: [1,2,3] · [2,2,2] = 12
```

**注意**: 这个程序需要CUDA驱动库支持，编译时需要链接`-lcuda`。

#### 源码技术实现要点

[`run_cubin.cpp`](run_cubin.cpp) 演示了动态kernel加载的完整技术流程。程序首先通过`cuModuleLoad()`动态加载指定的CUBIN文件，然后使用`cuModuleGetFunction()`获取kernel函数句柄。接下来程序会动态配置kernel的执行参数，包括内存分配、数据复制和kernel启动配置。整个过程展示了如何通过CUDA Driver API实现完整的资源管理和kernel执行。

**💡 重要技巧：避免C++名称修饰**

在动态加载CUBIN文件时，一个常见问题是C++编译器会对函数名进行名称修饰（name mangling），导致kernel名称变得复杂难读。例如：
- 原始名称：`vector_dot`
- 修饰后名称：`_Z10vector_dotPKfS0_Pfi`

**解决方案：使用extern "C"（推荐）**
```cpp
// 在kernel函数前添加extern "C"
extern "C" __global__ void vector_dot(const float *a, const float *b, float *result, int n) {
    // kernel代码
}
```

#### 实际应用场景

动态kernel加载技术在多个领域都有重要应用。在深度学习框架中，程序可以根据模型类型动态选择优化kernel；科学计算库能够根据数据类型选择精度最优的算法；图像处理应用可以根据图像特征选择最适合的滤波kernel；游戏引擎则能够根据场景复杂度动态选择渲染算法。这种技术让GPU计算变得更加智能和高效。



## CUDA执行模型

#### ✍️ Warp执行特性
- **执行模式**: 同一个warp内的线程执行相同的指令
- **分支处理**: 如果warp内线程执行不同分支，会导致warp分化(warp divergence)
- **性能影响**: warp分化会显著降低GPU的执行效率

**💡 重要概念回顾**：
Warp是GPU调度的基本单位，每个Warp包含32个线程，采用SIMT执行模型。关于Warp分化的详细解释和优化技巧，请参考前面的"✍️什么是Warp分化(Divergence)？"部分。

#### 线程块调度
- **SM分配**: 线程块被分配到不同的流式多处理器(SM)
- **资源限制**: 每个SM有固定的寄存器、共享内存和线程块数量
- **动态调度**: GPU自动管理线程块的调度和执行

#### 内存合并访问
- **概念**: 多个线程同时访问连续的内存地址
- **优化**: 合并访问可以提高内存带宽利用率
- **实现**: 使用合适的线程索引计算模式

**💡 性能优化提示**：
关于线程块调度优化、内存合并访问优化、共享内存使用等详细内容，将在Day 2的"性能分析和优化"部分深入讨论。


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

---

## 📁 相关文件快速链接
本教程包含以下相关程序文件，点击即可查看：

### 🚀 基础示例程序
- [`vector_add.cu`](vector_add.cu) - 向量加法CUDA kernel示例
- [`vector_dot.cu`](vector_dot.cu) - 向量点积CUDA kernel示例
- [`run_cubin.cpp`](run_cubin.cpp) - CUBIN文件运行程序（CUDA Driver API示例）

### 🎯 NCHW张量处理示例（不同维度Block对比）
- [`add_bias_nchw_1d.cu`](add_bias_nchw_1d.cu) - 1维Block版本，使用grid-stride loop处理4D张量
- [`add_bias_nchw_2d.cu`](add_bias_nchw_2d.cu) - 2维Block版本，threadIdx.x/y直接映射W/H维度
- [`add_bias_nchw_3d.cu`](add_bias_nchw_3d.cu) - 3维Block版本，threadIdx.x/y/z直接映射W/H/C维度

### 🔍 GPU信息获取工具
- [`gpu_info.py`](gpu_info.py) - Python版本GPU信息获取（推荐）
- [`gpu_info.cu`](gpu_info.cu) - CUDA版本完整硬件信息
- [`gpu_info_cpp.cpp`](gpu_info_cpp.cpp) - C++版本系统信息检查
- [`GPU_CONFIG_SUMMARY.md`](GPU_CONFIG_SUMMARY.md) - RTX 4090配置总结

### 📚 教程文档
- [`gbwt.md`](gbwt.md) - CUDA编程模型详解：Grid、Block、Thread、Warp的关系
- [`env.md`](env.md) - CUDA开发环境配置指南
