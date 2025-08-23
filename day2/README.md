# Day 2: 深入CUDA底层 - 手动加载PTX代码

## 概述
今天我们将深入了解CUDA的底层机制，学习如何使用CUDA Driver API手动加载PTX代码。这将帮助我们理解CUDA编译流程和运行时机制。

## 学习目标
- 理解CUDA编译流程：CUDA → PTX → CUBIN
- 掌握CUDA Driver API的基本使用
- 学会手动加载和执行PTX代码
- 理解CUDA Runtime API vs Driver API的区别

## CUDA编译流程详解

### 1. 编译阶段
```
CUDA源代码(.cu)
    ↓
PTX代码(.ptx) - 中间表示
    ↓
CUBIN文件(.cubin) - 二进制代码
    ↓
可执行文件
```

### 2. PTX (Parallel Thread Execution)
- PTX是CUDA的中间表示语言
- 类似于汇编语言，但更高级
- 可以跨GPU架构移植（需要重新编译）

### 3. CUBIN (CUDA Binary)
- 针对特定GPU架构的二进制代码
- 包含机器码和元数据
- 无法跨架构移植

## CUDA Runtime API vs Driver API

### Runtime API (高级接口)
- 更简单易用
- 自动内存管理
- 自动错误处理
- 适合大多数应用

### Driver API (低级接口)
- 更灵活，控制力更强
- 手动内存管理
- 手动错误处理
- 适合需要精细控制的场景

## 代码解析

### 1. PTX文件加载
```cpp
std::string loadPTX(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open PTX file!" << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
```

### 2. CUDA上下文初始化
```cpp
cuInit(0);                    // 初始化CUDA驱动
cuDeviceGet(&device, 0);      // 获取设备句柄
cuCtxCreate(&context, 0, device); // 创建CUDA上下文
```

### 3. PTX模块加载
```cpp
std::string ptxSource = loadPTX("vector_add.ptx");
CUresult res = cuModuleLoadDataEx(&module, ptxSource.c_str(), 0, nullptr, nullptr);
```

### 4. Kernel函数获取
```cpp
res = cuModuleGetFunction(&kernel, module, "vector_add");
```

### 5. 内存分配
```cpp
cuMemAlloc(&d_A, size);  // 分配GPU内存
cuMemAlloc(&d_B, size);
cuMemAlloc(&d_C, size);
```

### 6. Kernel启动
```cpp
res = cuLaunchKernel(
    kernel,                    // kernel函数句柄
    blocksPerGrid, 1, 1,      // Grid维度
    threadsPerBlock, 1, 1,    // Block维度
    0,                        // 共享内存大小
    0,                        // 流句柄
    args,                     // 参数数组
    nullptr                    // 额外参数
);
```

## 关键概念详解

### 1. CUDA上下文 (Context)
- 类似于OpenGL的渲染上下文
- 管理GPU资源（内存、模块等）
- 一个进程可以有多个上下文

### 2. CUDA模块 (Module)
- 包含编译后的代码和符号
- 可以动态加载和卸载
- 支持热更新

### 3. 内存管理
- `cuMemAlloc`: 分配GPU内存
- `cuMemcpyHtoD`: 主机到设备内存复制
- `cuMemcpyDtoH`: 设备到主机内存复制
- `cuMemFree`: 释放GPU内存

## 编译和运行

### 编译命令
```bash
nvcc -o run_ptx_manual run_ptx_manual.cu -lcuda
```

**注意**: 需要链接`libcuda`库

### 运行命令
```bash
./run_ptx_manual
```

## 性能分析

### Driver API的优势
- 更低的启动开销
- 更精细的内存控制
- 支持异步操作

### 适用场景
- 需要动态加载代码
- 需要精细控制内存
- 构建CUDA运行时库

## 常见问题和调试技巧

### 1. PTX加载失败
- 检查PTX文件是否存在
- 验证PTX语法是否正确
- 使用`cuGetErrorString`获取详细错误信息

### 2. 内存分配失败
- 检查GPU内存是否充足
- 验证内存大小是否合理

### 3. Kernel启动失败
- 检查线程配置是否有效
- 验证参数类型和数量

## 高级特性

### 1. 动态代码生成
- 运行时生成PTX代码
- 支持JIT编译
- 实现动态优化

### 2. 多GPU支持
- 每个GPU一个上下文
- 支持GPU间数据传输
- 负载均衡

### 3. 异步操作
- 使用CUDA流
- 重叠计算和传输
- 提高整体性能

## 下一步
明天我们将学习矩阵乘法，这是深度学习中的核心操作，也是理解CUDA性能优化的绝佳例子。

## 练习
1. 修改代码支持不同的PTX文件
2. 添加错误处理和日志记录
3. 实现异步内存传输
4. 尝试加载不同的kernel函数

## 参考资料
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
