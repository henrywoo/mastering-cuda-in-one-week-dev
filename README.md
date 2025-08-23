# CUDA编程教程系列 - 从入门到精通

## 概述

这是一个完整的CUDA编程教程系列，专为初级SDE设计，循序渐进地教授CUDA编程的核心概念和实践技巧。通过7天的学习，你将掌握从基础概念到高级优化的完整知识体系，包括最新的LLM优化技术和不同GPU架构的优化策略。

## 教程结构

每个Day的教程都包含完整的理论讲解、代码示例、实践项目和"相关文件快速链接"section，方便读者快速找到相关代码文件和学习资源。

### Day 1: CUDA编程基础 - 硬件架构与编程模型
- **学习目标**: 理解CUDA编程模型，掌握GPU硬件架构和内存层次结构
- **核心概念**: 线程层次结构、Warp执行特性、内存管理、动态kernel加载
- **实践项目**: 向量加法、向量点积、GPU配置工具、CUBIN文件运行
- **文件**: `day1/README.md`, `day1/vector_add.cu`, `day1/vector_dot.cu`, `day1/run_cubin.cpp`, `day1/gpu_info.py`

### Day 2: CUDA调试与优化 - PTX加载与性能分析
- **学习目标**: 理解CUDA编译流程，掌握Driver API，学会调试和性能优化
- **核心概念**: PTX、CUBIN、CUDA上下文、性能分析工具、优化策略
- **实践项目**: 手动加载和执行PTX代码、性能分析和优化实战
- **文件**: `day2/README.md`, `day2/run_ptx_manual.cu`

### Day 3: 矩阵乘法优化 - CUDA性能调优实战
- **学习目标**: 掌握矩阵乘法的CUDA实现和优化
- **核心概念**: 共享内存、内存合并访问、CUDA流
- **NVIDIA库**: cuBLAS、CUTLASS介绍和使用
- **实践项目**: 多种优化版本的矩阵乘法
- **文件**: `day3/README.md`, `day3/matrix_mul.cu`

### Day 4: 卷积神经网络(CNN) - CUDA深度学习实战
- **学习目标**: 实现CNN核心操作，理解卷积优化
- **核心概念**: 2D卷积、共享内存优化、分离卷积
- **NVIDIA库**: cuDNN库介绍和性能对比
- **实践项目**: 多种卷积算法的CUDA实现
- **文件**: `day4/README.md`, `day4/cnn_conv.cu`

### Day 5: 注意力机制和Transformer - 现代NLP的CUDA实现
- **学习目标**: 掌握注意力机制和Transformer架构
- **核心概念**: 自注意力、多头注意力、位置编码
- **实践项目**: 完整的Transformer实现
- **文件**: `day5/README.md`

### Day 6: 最新LLM CUDA Kernel定制优化 - 前沿技术实战
- **学习目标**: 掌握最新的LLM优化技术
- **核心概念**: Flash Attention、Paged Attention、Grouped Query Attention
- **前沿技术**: 稀疏注意力、最新Tensor Core优化
- **实践项目**: 多种注意力优化算法的实现
- **文件**: `day6/README.md`

### Day 7: CUDA性能调优高级技巧 - 从理论到实践
- **学习目标**: 掌握高级性能调优技巧
- **核心概念**: 内存层次优化、指令级优化、架构特定优化
- **GPU架构**: 不同架构特性对比，包括最新的Blackwell架构
- **实践项目**: 性能分析和优化实战
- **文件**: `day7/README.md`, `day7/performance_tuning.cu`

## 学习路径

```
基础概念和硬件架构 (Day 1)
    ↓
调试技巧和性能优化基础 (Day 2)
    ↓
算法实现和优化 (Day 3-4)
    ↓
深度学习模型 (Day 5-6)
    ↓
高级性能调优技巧 (Day 7)
```

## 前置要求

### 硬件要求
- NVIDIA GPU (支持CUDA)
- 推荐: GTX 1060或更高版本
- 最新架构: RTX 40系列、H100、B100等

### 软件要求
- CUDA Toolkit 11.0+
- 支持C++14的编译器
- Linux/Windows/macOS

### 基础知识
- C/C++编程基础
- 基本的并行计算概念
- 线性代数基础

## 快速开始

### 1. 环境准备
```bash
# 安装CUDA Toolkit
# 下载并安装适合你系统的CUDA版本

# 验证安装
nvcc --version
nvidia-smi
```

### 2. 编译第一个程序
```bash
cd day1
nvcc -o vector_add vector_add.cu
./vector_add
```

### 3. 学习建议
- 每天花2-3小时学习
- 动手实践每个代码示例
- 尝试修改参数观察效果
- 使用性能分析工具
- 对比不同GPU架构的性能差异

## 核心概念速查

### CUDA编程模型
- **Host**: CPU端代码，负责内存分配、数据传输、kernel启动
- **Device**: GPU端代码，执行并行计算任务
- **Kernel**: GPU上执行的函数，使用`__global__`修饰符
- **Grid**: 线程块的集合，定义整个计算任务的规模
- **Block**: 线程的集合，线程块内的线程可以协作和共享内存
- **Thread**: 最小的执行单元，每个线程处理一个数据元素
- **Warp**: 32个线程的执行组，GPU调度的基本单位

### 内存层次
- **Registers**: 寄存器，最快，容量最小，每个线程私有
- **Shared Memory**: 共享内存，容量小，延迟低，线程块内共享
- **L1 Cache**: 一级缓存，自动管理，减少全局内存访问
- **L2 Cache**: 二级缓存，全局共享，提高内存访问效率
- **Global Memory**: 全局内存，容量大，延迟高，所有线程共享
- **Constant Memory**: 常量内存，只读，有缓存机制
- **Texture Memory**: 纹理内存，针对空间局部性优化

### 优化策略
- **内存合并访问**: 相邻线程访问相邻内存，提高内存带宽利用率
- **共享内存使用**: 缓存频繁访问的数据，减少全局内存访问
- **循环展开**: 减少循环开销，提高指令级并行性
- **分支优化**: 减少Warp分化，保持线程执行一致性
- **寄存器优化**: 合理使用寄存器，避免寄存器溢出
- **线程块配置**: 选择最优的线程块大小，平衡资源利用
- **异步操作**: 使用CUDA流重叠计算和内存传输

## NVIDIA官方库介绍

### cuBLAS
- **用途**: 基础线性代数运算
- **优势**: 高度优化的矩阵乘法、向量运算
- **适用场景**: 科学计算、机器学习

### CUTLASS
- **用途**: 可定制的线性代数库
- **优势**: 支持自定义算法、多种精度
- **适用场景**: 需要特殊优化的应用

### cuDNN
- **用途**: 深度学习加速库
- **优势**: 优化的卷积、池化、归一化操作
- **适用场景**: CNN、RNN等神经网络

## GPU架构特性

### 最新架构对比
- **Ampere (RTX 30, A100)**: 第三代Tensor Core, 动态并行
- **Hopper (H100, H200)**: 第四代Tensor Core, Transformer Engine
- **Ada Lovelace (RTX 40)**: 第四代Tensor Core, DLSS 3.0
- **Blackwell (B100, B200)**: 第五代Tensor Core, tcgen05.mma指令

### 架构特定优化
- **Tensor Core**: 混合精度矩阵乘法加速
- **RT Core**: 光线追踪加速
- **专用引擎**: AI推理、视频编码等

## 性能分析工具

### 内置工具
- `nvprof`: 性能分析器，命令行性能分析
- `nvidia-smi`: GPU监控，实时状态查看
- `cuda-memcheck`: 内存检查，检测内存错误
- `compute-sanitizer`: 运行时错误检测，替代cuda-memcheck
- `cuda-gdb`: CUDA调试器，支持断点和变量查看

### Nsight工具
- **Nsight Systems**: 系统级性能分析
- **Nsight Compute**: Kernel级性能分析
- **Nsight Graphics**: 图形调试

## 常见问题

### 编译问题
```bash
# 常见编译错误
nvcc: command not found  # 安装CUDA Toolkit
error: identifier "__global__" is undefined  # 包含cuda_runtime.h
```

### 运行时问题
```bash
# 内存不足
cudaErrorMemoryAllocation  # 减少数据大小或批处理大小

# Kernel启动失败
cudaErrorInvalidConfiguration  # 检查线程块配置
```

### 性能问题
- 内存带宽瓶颈: 使用共享内存
- 计算瓶颈: 优化算法或使用Tensor Core
- 同步开销: 减少kernel间同步

## 进阶学习

### 库和框架
- **cuBLAS**: 基础线性代数
- **cuDNN**: 深度学习加速
- **Thrust**: 并行算法库
- **CUB**: 并行原语库

### 高级特性
- **Tensor Core**: 混合精度计算
- **Multi-GPU**: 多GPU编程
- **Unified Memory**: 统一内存管理
- **Dynamic Parallelism**: 动态并行

### 实际应用
- 科学计算
- 图像处理
- 机器学习
- 金融计算

## 最新技术趋势

### LLM优化技术
- **Flash Attention**: 内存高效的注意力机制
- **Paged Attention**: 分页管理的KV缓存
- **Grouped Query Attention**: 分组查询优化
- **稀疏注意力**: 减少计算量的注意力模式

### 硬件优化
- **新一代Tensor Core**: 更高的计算密度
- **HBM3e内存**: 5TB/s带宽
- **专用AI引擎**: 针对大模型优化

## 贡献和反馈

如果你发现教程中的错误或有改进建议，欢迎：

1. 提交Issue
2. 创建Pull Request
3. 发送邮件反馈

## 许可证

本教程采用MIT许可证，你可以自由使用、修改和分发。

## 致谢

感谢NVIDIA提供的优秀CUDA平台和工具，以及开源社区的支持。

---

**开始你的CUDA编程之旅吧！** 🚀

记住：实践是最好的老师，多写代码，多调试，多优化！
