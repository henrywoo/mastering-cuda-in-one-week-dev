# Day 1: CUDA Programming Basics - Hardware Architecture and Programming Model

## Overview
Today we will begin our CUDA programming journey, starting with the fundamental concepts of GPU hardware architecture and CUDA programming model. By understanding GPU hardware characteristics, memory hierarchy, and thread execution model, we will lay a solid foundation for subsequent CUDA programming. We will practice these concepts through vector addition and vector dot product examples.

## Learning Objectives
- Understand basic concepts of CUDA programming model (Host vs Device, Kernel, Grid, Block, Thread)
- Master GPU hardware architecture and memory hierarchy (SM, Warp, registers, shared memory, etc.)
- Learn to write simple CUDA kernel functions (vector addition, vector dot product)
- Understand thread hierarchy and index calculation (Grid-Block-Thread relationship)
- Master basic CUDA memory management (cudaMalloc, cudaMemcpy, etc.)
- Learn to use GPU configuration tools to obtain hardware parameters and optimization suggestions
- Understand dynamic kernel loading mechanism (CUBIN files, Driver API)
- Master Warp execution characteristics and basic methods to avoid Warp divergence

## Development Environment
This tutorial is primarily based on **NVIDIA GPU** explanations. The author's development environment uses:
- **Operating System**: Linux Ubuntu 22.04 LTS
- **GPU**: NVIDIA GeForce RTX 4090 (Ada Lovelace architecture)
- **CUDA Version**: CUDA 12.4
- **Compiler**: nvcc (NVIDIA CUDA Compiler)

> 📋 **Environment Setup**: If you haven't installed the CUDA development environment(including `nvcc`) yet, please refer to the [Environment Setup Guide](./env.md) for installation and configuration.

Although different GPU models may vary in specific parameters, the basic concepts and APIs of CUDA programming are consistent.

## GPU Hardware Basics

### GPU vs CPU Architecture Differences

#### CPU Architecture Characteristics
CPU (Central Processing Unit) adopts a serial execution architecture, equipped with a small number of powerful computing cores. Each core has complex control logic and branch prediction capabilities, able to intelligently predict program execution paths and achieve out-of-order execution, thereby maximizing instruction-level parallelism. CPUs have a massive multi-level cache system, including L1, L2, L3 caches, which can effectively reduce memory access latency and improve data locality. Since CPU's design goal is universality, it can efficiently handle various types of computing tasks, from complex control logic to simple arithmetic operations.

#### GPU Architecture Characteristics
GPU (Graphics Processing Unit) adopts a parallel execution architecture, equipped with a large number of relatively simple computing cores. Although individual cores are not as powerful as CPU cores, there are many of them, capable of simultaneously processing large numbers of similar parallel tasks. GPUs use the SIMT (Single Instruction, Multiple Thread) execution model, where multiple threads execute the same instruction but process different data. This design is particularly suitable for data-parallel compute-intensive tasks. In terms of memory systems, GPUs adopt a hierarchical memory design, including global memory, shared memory, registers, and other levels, each with different access latency and capacity characteristics. Since GPU's design goal is clear, it excels in parallel computing fields such as image processing, scientific computing, and deep learning, but is less efficient than CPUs in complex serial tasks.

### GPU Hardware Architecture and Memory Layout

Modern GPUs adopt a layered architecture design, which can be divided into three main levels from macro to micro. The top level is the overall GPU device, containing multiple Streaming Multiprocessors (SM), which are the core computing units of the GPU. Each SM has independent instruction scheduling and execution capabilities. Below the SM level is the shared L2 cache, providing unified data caching services for all SMs. The bottom level is global memory, usually using HBM (High Bandwidth Memory) or GDDR technology, providing large-capacity, high-bandwidth storage space for the entire GPU.

Each SM internally adopts fine-grained parallel design. At the top level of the SM, multiple Warps execute in parallel, sharing the SM's computing resources.

> **Warp Concept Details**:
A Warp is the basic unit of GPU scheduling, with each Warp containing 32 threads. Warps use the SIMT (Single Instruction, Multiple Thread) execution model, meaning all threads within the same Warp execute the same instruction but process different data. This design allows GPUs to efficiently utilize data parallelism. When multiple threads need to execute the same operation, only one instruction is needed to control 32 threads executing simultaneously. **🎯 Important Note**: Warp size 32 is a fixed design of NVIDIA GPU architecture, unchanged from the earliest Tesla architecture to the latest Blackwell architecture. The number 32 is carefully designed to fully utilize GPU's SIMT execution units.

Below the Warp level, SMs are equipped with three key memory resources: Register File, Shared Memory, and L1 Cache. The register file provides the fastest storage access for each thread, shared memory supports data exchange and collaboration within thread blocks, and L1 cache provides an additional data caching layer. At the bottom level of the SM, various dedicated functional units are configured, including FP32, FP64 floating-point arithmetic units, integer arithmetic units, and Tensor Cores and other dedicated accelerators.

GPU memory systems adopt a hierarchical design, from fastest to slowest access speed: registers, shared memory, local memory, constant memory, texture memory, and global memory. Registers are private storage space for each thread, with access latency of only 1 clock cycle, but limited capacity, with each thread able to use at most 255 32-bit registers. Shared memory is storage space shared within thread blocks, with access latency of 1-2 clock cycles, capacity of 48KB per thread block, suitable for storing frequently accessed intermediate results and implementing thread collaboration. Local memory, although nominally thread-private, is actually stored in global memory, with high access latency (200-800 clock cycles), mainly used for storing large arrays and register overflow data. Constant memory has caching mechanisms, suitable for storing kernel parameters and lookup tables, performing best when multiple threads access the same address. Texture memory is optimized for 2D/3D spatial locality access, suitable for image processing and scientific computing applications. Global memory has the largest capacity but highest access latency, serving as the main storage space shared by all threads. Its performance largely depends on memory access patterns, with coalesced access significantly improving memory bandwidth utilization.

#### GPU Overall Architecture Diagram
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

Modern GPUs adopt a layered architecture design, which can be divided into three main levels from macro to micro. The top level is the overall GPU device, containing multiple Streaming Multiprocessors (SM), which are the core computing units of the GPU. Each SM has independent instruction scheduling and execution capabilities. Below the SM level is the shared L2 cache, providing unified data caching services for all SMs. The bottom level is global memory, usually using HBM (High Bandwidth Memory) or GDDR technology, providing large-capacity, high-bandwidth storage space for the entire GPU.

#### Single SM Internal Structure Diagram
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

Each SM internally adopts fine-grained parallel design. At the top level of the SM, multiple Warps execute in parallel, sharing the SM's computing resources.

> **Warp Concept Details**:
A Warp is the basic unit of GPU scheduling, with each Warp containing 32 threads. Warps use the SIMT (Single Instruction, Multiple Thread) execution model, meaning all threads within the same Warp execute the same instruction but process different data. This design allows GPUs to efficiently utilize data parallelism. When multiple threads need to execute the same operation, only one instruction is needed to control 32 threads executing simultaneously. **🎯 Important Note**: Warp size 32 is a fixed design of NVIDIA GPU architecture, unchanged from the earliest Tesla architecture to the latest Blackwell architecture. The number 32 is carefully designed to fully utilize GPU's SIMT execution units.

Below the Warp level, SMs are equipped with three key memory resources: Register File, Shared Memory, and L1 Cache. The register file provides the fastest storage access for each thread, shared memory supports data exchange and collaboration within thread blocks, and L1 cache provides an additional data caching layer. At the bottom level of the SM, various dedicated functional units are configured, including FP32, FP64 floating-point arithmetic units, integer arithmetic units, and Tensor Cores and other dedicated accelerators.

#### GPU Memory Hierarchy Diagram
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
│  │   Constant    │  │   Texture     │  │    Global     │          │
│  │    Memory     │  │    Memory     │  │    Memory     │          │
│  │   (Cached)    │  │  (Optimized)  │  │   (Slowest)   │          │
│  │               │  │               │  │               │          │
│  └───────────────┘  └───────────────┘  └───────────────┘          │
└───────────────────────────────────────────────────────────────────┘
```

GPU memory systems adopt a hierarchical design, from fastest to slowest access speed: registers, shared memory, local memory, constant memory, texture memory, and global memory. Registers are private storage space for each thread, with access latency of only 1 clock cycle, but limited capacity, with each thread able to use at most 255 32-bit registers. Shared memory is storage space shared within thread blocks, with access latency of 1-2 clock cycles, capacity of 48KB per thread block, suitable for storing frequently accessed intermediate results and implementing thread collaboration. Local memory, although nominally thread-private, is actually stored in global memory, with high access latency (200-800 clock cycles), mainly used for storing large arrays and register overflow data. Constant memory has caching mechanisms, suitable for storing kernel parameters and lookup tables, performing best when multiple threads access the same address. Texture memory is optimized for 2D/3D spatial locality access, suitable for image processing and scientific computing applications. Global memory has the largest capacity but highest access latency, serving as the main storage space shared by all threads. Its performance largely depends on memory access patterns, with coalesced access significantly improving memory bandwidth utilization.

## CUDA Programming Model Basics

The best way to learn CUDA programming is through a concrete example. Let's first look at a complete vector addition program to get a feel for what CUDA code looks like:

```cuda
// CUDA kernel function for vector addition
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host code calling GPU kernel
int main() {
    // ... memory allocation and data preparation ...
    
    // Launch kernel, configure thread grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // ... result retrieval and cleanup ...
}
```

Looking at this code, you might have many questions: What is `__global__`? What does the strange `<<<>>>` syntax mean? Why do we need to calculate `blocksPerGrid`? These are exactly the core concepts we need to learn.

We chose vector addition as our introductory example precisely because it's simple enough to let us focus on understanding CUDA's programming model without being distracted by complex algorithmic logic. Through this example, we will step by step enter the world of CUDA, understanding its design philosophy and operational mechanisms.

## Understanding CUDA's Dual World

Returning to our vector addition program, you might notice that the code is divided into two parts: one is the main function running on the CPU, and the other is the kernel function executing on the GPU. This is the core idea of CUDA programming—**separation of concerns**.

### Host Code: CPU's Command Center

In CUDA programming, the CPU plays the role of a "command center." It's responsible for controlling the overall program flow, like a conductor of an orchestra—not directly playing instruments, but coordinating the rhythm and sequence of the entire performance.

The code on the CPU (which we call host code) is written in standard C++. Its main tasks include:
- Preparing data, like a chef preparing ingredients
- Allocating space in CPU memory, preparing "temporary storage points" for data
- Allocating space in GPU memory, like preparing a "workbench" for the GPU
- Launching GPU kernels, equivalent to giving the GPU the "start working" command
- Retrieving calculation results from the GPU, like collecting completed works
- Processing results and output, doing final organization and presentation

### Device Code: GPU's Parallel Factory

The code on the GPU (device code) focuses on computation itself. It uses CUDA C++ extension syntax, telling the compiler through keywords like `__global__`: "This code should run on the GPU."

The GPU is like a massive parallel factory, where each thread is a small worker, simultaneously processing different data. In our vector addition, each thread is responsible for calculating the sum of one element, so we can process thousands of elements simultaneously, rather than calculating one by one like the CPU.

## The Art of Thread Organization

Now let's deeply understand how the GPU organizes these "small workers." Imagine if you were managing a large factory, how would you organize the workers?

### Understanding Thread Hierarchy Through Factory Management

The GPU adopts a three-layer management structure, just like a modern factory management system:

**First Layer: Factory Overall (Grid)**
The entire GPU is like a large factory, which we call Grid. This factory can run multiple production workshops simultaneously, each workshop responsible for different tasks. In our vector addition, the Grid is the total collection of all elements that need to be calculated.

**Second Layer: Production Workshops (Block)**
Each production workshop (Block) has a fixed number of workers who can collaborate with each other and share tools and materials. Threads within a Block can access the same shared memory, just like workers in a workshop can share a toolbox.

**Third Layer: Workers (Thread)**
Each worker (Thread) is responsible for specific computational tasks. In our example, each thread calculates the sum of one vector element. Although all threads execute the same code, they process different data.

```
Factory (Grid)
├── Workshop 0 (Block 0)
│   ├── Worker 0 (Thread 0)
│   ├── Worker 1 (Thread 1)
│   └── ...
├── Workshop 1 (Block 1)
│   ├── Worker 0 (Thread 0)
│   ├── Worker 1 (Thread 1)
│   └── ...
└── ...
```

### Assigning Tasks to Each Worker

Now the question is: How does each worker know which data they should process? This is the core of thread index calculation.

In our vector addition, each thread needs to calculate:
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

This formula is like assigning work numbers to each worker in the factory. Let me explain it with a more relatable example from life:

## Inspiration from School Roll Call System

Imagine a school roll call system. Suppose the school has 3 classes, each class has 4 students, and you need to number all students in the school. How would you design it?

**Class Structure**:
- Class 0: Student 0 (ID 0), Student 1 (ID 1), Student 2 (ID 2), Student 3 (ID 3)
- Class 1: Student 0 (ID 4), Student 1 (ID 5), Student 2 (ID 6), Student 3 (ID 7)  
- Class 2: Student 0 (ID 8), Student 1 (ID 9), Student 2 (ID 10), Student 3 (ID 11)

**ID Calculation Formula**:
```
Student ID = Class Number × Students per Class + Student Number in Class
```

For example:
- Class 1, Student 2: 1 × 4 + 2 = 6
- Class 2, Student 1: 2 × 4 + 1 = 9

**Correspondence in CUDA**:
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
- `blockIdx.x` = Class Number (0, 1, 2...)
- `blockDim.x` = Students per Class (4)
- `threadIdx.x` = Student Number in Class (0, 1, 2, 3)
- `idx` = Global Student ID

This way, each thread knows which element in the array it should process. Through this clever index design, the GPU can efficiently distribute work, giving each thread a clear task, thereby achieving true parallel computation.

## Kernel Functions

### Vector Addition Kernel

```cuda
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**Kernel Launch**:
```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
```

### Vector Dot Product Kernel

```cuda
__global__ void vector_dot(const float *a, const float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        atomicAdd(result, a[idx] * b[idx]);
    }
}
```

## Compilation Commands

### Basic Compilation
```bash
nvcc -o vector_add vector_add.cu
nvcc -o vector_dot vector_dot.cu
```

### Optimization Flags
```bash
nvcc -O3 -arch=sm_89 -o vector_add vector_add.cu
nvcc -O3 -arch=sm_89 -o vector_dot vector_dot.cu
```

### Generate PTX and CUBIN
```bash
nvcc -ptx -o vector_add.ptx vector_add.cu
nvcc -cubin -o vector_add.cubin vector_add.cu
```

## CUDA Execution Model

### Warp Execution Characteristics

**✍️ What is Warp Divergence (Divergence)?**
Warp divergence occurs when threads within the same Warp take different execution paths due to conditional statements (if-else, switch, etc.). When divergence happens, the GPU must execute both paths sequentially, significantly reducing performance.

**Example of Warp Divergence**:
```cuda
__global__ void divergent_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (data[idx] > 0) {
            data[idx] = data[idx] * 2;  // Path A
        } else {
            data[idx] = data[idx] / 2;  // Path B
        }
    }
}
```

**Impact of Divergence**:
- Performance degradation: 2x to 32x slower
- Resource waste: Some threads are idle
- Reduced parallelism: Sequential execution instead of parallel

**How to Avoid Divergence**:
1. **Data Reorganization**: Sort data to group similar values
2. **Algorithm Redesign**: Use branch-free algorithms
3. **Conditional Compilation**: Use template parameters
4. **Predication**: Use conditional assignment instead of conditional execution

**💡 Important Concept Review**:
For detailed explanations of Warp concepts and divergence, see the earlier sections:
- [Warp Concept Details](#gpu硬件架构和内存布局)
- [✍️ What is Warp Divergence (Divergence)?](#✍️-warp执行特性)

**💡 Performance Optimization Tips**:
For detailed optimization strategies and performance analysis tools, see [Day 2: Performance Optimization](day2/README.md).

## Memory Management

### Basic Memory Operations

```cuda
// Memory allocation
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, n * sizeof(float));
cudaMalloc(&d_b, n * sizeof(float));
cudaMalloc(&d_c, n * sizeof(float));

// Data transfer
cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

// Result retrieval
cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

// Memory cleanup
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

### Error Handling

```cuda
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return -1;
}
```

## GPU Configuration Tools

### 💡 GPU Parameter Tool

To get detailed GPU configuration and optimal thread block size, we provide three versions of GPU information tools:

**Python Version (Recommended)**:
- [gpu_info.py](gpu_info.py) - Easy to use, no compilation required
- Shows GPU name, compute capability, memory, SM count, etc.
- Provides optimal thread block size recommendations

**CUDA C++ Version**:
- [gpu_info.cu](gpu_info.cu) - Full GPU information using CUDA API
- Compile with: `nvcc -o gpu_info gpu_info.cu`
- Shows detailed hardware specifications

**Pure C++ Version**:
- [gpu_info_cpp.cpp](gpu_info_cpp.cpp) - System information and CUDA environment check
- Compile with: `g++ -o gpu_info_cpp gpu_info_cpp.cpp`
- Checks CUDA installation and environment variables

**Configuration Summary**:
- [GPU_CONFIG_SUMMARY.md](GPU_CONFIG_SUMMARY.md) - RTX 4090 configuration summary
- Contains optimal CUDA programming recommendations
- Based on `nvidia-smi -q` output analysis

**Recommended Usage**:
1. **Quick Check**: Use `gpu_info.py` for immediate GPU information
2. **Detailed Analysis**: Use `gpu_info.cu` for comprehensive hardware details
3. **Environment Verification**: Use `gpu_info_cpp.cpp` to check CUDA setup
4. **Reference**: Use `GPU_CONFIG_SUMMARY.md` for optimization guidelines

## Dynamic Kernel Loading System

### What are CUBIN Files?

**CUBIN (CUDA Binary)**: Binary files containing compiled GPU machine code, generated by the NVIDIA compiler from CUDA source code.

**Generation Process**:
1. **CUDA Source (.cu)** → **PTX (.ptx)** → **CUBIN (.cubin)**
2. **nvcc** compiles CUDA source to PTX (intermediate representation)
3. **ptxas** assembles PTX to CUBIN (final binary)

**CUBIN vs PTX**:
- **CUBIN**: Binary format, faster loading, smaller size
- **PTX**: Text format, human-readable, larger size
- **CUBIN** is preferred for production use

### 🚀 Dynamic Kernel Loading System

The dynamic kernel loading system demonstrates how to load and execute CUDA kernels at runtime using the CUDA Driver API. This approach provides flexibility and enables plugin-based architectures.

**Key Features**:
- **Runtime Loading**: Load kernels without recompiling the main program
- **Multiple Kernels**: Support different kernel functions in the same program
- **Error Handling**: Comprehensive error checking and resource management
- **RAII Design**: Modern C++ resource management using RAII principles

**Implementation Highlights**:
```cpp
// RAII wrapper for CUDA resources
class CUDADevice {
    // Automatic resource management
    // Exception-safe initialization and cleanup
};

class CUDAModule {
    // CUBIN file loading and management
    // Automatic module unloading
};

class CUDAMemory {
    // Device memory allocation and deallocation
    // Automatic memory cleanup
};
```

**💡 Important Technique: Avoiding C++ Name Mangling**

**Problem**: C++ compilers perform name mangling, changing function names like `vector_add` to `_Z10vector_addPKfS0_Pfi`.

**Solution 1 (Recommended)**: Use `extern "C"` in kernel source files
```cuda
extern "C" __global__ void vector_add(const float *a, const float *b, float *c, int n) {
    // kernel implementation
}
```

**Benefits**:
- Simple and effective
- No additional compilation flags needed
- Maintains clean, readable kernel names
- Compatible with all CUDA versions

**Usage Examples**:
```bash
# Load and execute vector addition
./run_cubin vector_add.cubin    # Execute vector addition: [1,2,3] + [2,2,2] = [3,4,5]

# Load and execute vector dot product  
./run_cubin vector_dot.cubin    # Execute vector dot product: [1,2,3] · [2,2,2] = 12
```

## Quick Start

### 1. Environment Setup
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Check GPU information
python3 gpu_info.py
```

### 2. Compile and Run
```bash
# Vector addition
nvcc -o vector_add vector_add.cu
./vector_add

# Vector dot product
nvcc -o vector_dot vector_dot.cu
./vector_dot

# Dynamic kernel loading
nvcc -arch=sm_89 run_cubin.cpp -lcuda -o run_cubin
./run_cubin vector_add.cubin
./run_cubin vector_dot.cubin
```

### 3. Generate PTX and CUBIN
```bash
# Generate PTX (intermediate representation)
nvcc -ptx -o vector_add.ptx vector_add.cu

# Generate CUBIN (binary)
nvcc -cubin -o vector_add.cubin vector_add.cu

# View CUBIN contents
cuobjdump -sass vector_add.cubin
```

## Summary

Today we have learned:
1. **GPU Hardware Architecture**: Understanding SM, Warp, memory hierarchy
2. **CUDA Programming Model**: Grid-Block-Thread relationship and index calculation
3. **Basic Kernels**: Vector addition and dot product implementation
4. **Memory Management**: Host vs device memory, allocation and transfer
5. **GPU Tools**: Configuration tools and parameter optimization
6. **Dynamic Loading**: CUBIN files and Driver API usage
7. **Warp Characteristics**: Execution model and divergence avoidance

**Key Concepts**:
- **Warp Size**: Fixed at 32 threads across all NVIDIA GPU architectures
- **Thread Block Size**: 256 is a balanced choice for most applications
- **Memory Hierarchy**: Registers → Shared Memory → Global Memory
- **Index Calculation**: `idx = blockIdx.x * blockDim.x + threadIdx.x`

**Next Steps**:
- Practice with different data sizes and thread configurations
- Experiment with memory access patterns
- Explore advanced optimization techniques in Day 2

## 📁 Quick File Links

**Main Files**:
- [README.md](README.md) - This tutorial file
- [vector_add.cu](vector_add.cu) - Vector addition kernel
- [vector_dot.cu](vector_dot.cu) - Vector dot product kernel
- [run_cubin.cpp](run_cubin.cpp) - Dynamic kernel loading system

**GPU Information Tools**:
- [gpu_info.py](gpu_info.py) - Python GPU configuration tool
- [gpu_info.cu](gpu_info.cu) - CUDA C++ GPU information tool
- [gpu_info_cpp.cpp](gpu_info_cpp.cpp) - Pure C++ system information tool
- [GPU_CONFIG_SUMMARY.md](GPU_CONFIG_SUMMARY.md) - RTX 4090 configuration summary

**Generated Files**:
- [vector_add.ptx](vector_add.ptx) - PTX intermediate representation
- [vector_add.cubin](vector_add.cubin) - CUBIN binary file
- [vector_dot.ptx](vector_dot.ptx) - PTX intermediate representation
- [vector_dot.cubin](vector_dot.cubin) - CUBIN binary file

**Compilation Commands**:
```bash
# Basic compilation
nvcc -o vector_add vector_add.cu
nvcc -o vector_dot vector_dot.cu

# With optimization
nvcc -O3 -arch=sm_89 -o vector_add vector_add.cu
nvcc -O3 -arch=sm_89 -o vector_dot vector_dot.cu

# Generate PTX and CUBIN
nvcc -ptx -o vector_add.ptx vector_add.cu
nvcc -cubin -o vector_add.cubin vector_add.cu

# Dynamic kernel loading
nvcc -arch=sm_89 run_cubin.cpp -lcuda -o run_cubin
```
