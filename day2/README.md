# Day 2: CUDA Debugging and Optimization - PTX Loading and Performance Analysis

## Overview
Today we will dive deep into CUDA debugging and optimization, learning how to use CUDA Driver API to manually load PTX code, master CUDA program debugging techniques and performance analysis methods. By understanding CUDA compilation process, runtime mechanisms, and using professional performance analysis tools, we will learn how to diagnose performance bottlenecks and apply optimization strategies.

## Learning Objectives
- Understand CUDA compilation process: CUDA → PTX → CUBIN
- Master basic usage of CUDA Driver API (cuInit, cuDeviceGet, cuCtxCreate, etc.)
- Learn to manually load and execute PTX code (cuModuleLoadDataEx, cuLaunchKernel, etc.)
- Understand differences and use cases between CUDA Runtime API vs Driver API
- Master CUDA program debugging techniques (error handling, memory checking, boundary validation, etc.)
- Learn to use performance analysis tools (nvprof, Nsight Systems, Nsight Compute)
- Understand and apply performance optimization strategies (memory coalescing, shared memory usage, register optimization, etc.)
- Master Warp divergence optimization and thread block configuration optimization techniques

## CUDA Compilation Process Details

### 1. Compilation Stages
```
CUDA Source Code (.cu)
    ↓
PTX Code (.ptx) - Intermediate Representation
    ↓
CUBIN File (.cubin) - Binary Code
    ↓
Executable File
```

### 2. PTX (Parallel Thread Execution)
- PTX is CUDA's intermediate representation language
- Similar to assembly language but more advanced
- Can be ported across GPU architectures (requires recompilation)

### 3. CUBIN (CUDA Binary)
- Binary code targeting specific GPU architectures
- Contains machine code and metadata
- Cannot be ported across architectures

## CUDA Runtime API vs Driver API

### Runtime API (High-Level Interface)
- Simpler and easier to use
- Automatic memory management
- Automatic error handling
- Suitable for most applications

### Driver API (Low-Level Interface)
- More flexible, stronger control
- Manual memory management
- Manual error handling
- Suitable for scenarios requiring fine-grained control

## Code Analysis

### 1. PTX File Loading
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

### 2. CUDA Context Initialization
```cpp
cuInit(0);                    // Initialize CUDA driver
cuDeviceGet(&device, 0);      // Get device handle
cuCtxCreate(&context, 0, device); // Create CUDA context
```

### 3. PTX Module Loading
```cpp
std::string ptxSource = loadPTX("vector_add.ptx");
CUresult res = cuModuleLoadDataEx(&module, ptxSource.c_str(), 0, nullptr, nullptr);
```

### 4. Kernel Function Retrieval
```cpp
res = cuModuleGetFunction(&kernel, module, "vector_add");
```

### 5. Memory Allocation
```cpp
cuMemAlloc(&d_A, size);  // Allocate GPU memory
cuMemAlloc(&d_B, size);
cuMemAlloc(&d_C, size);
```

### 6. Kernel Launch
```cpp
res = cuLaunchKernel(
    kernel,                    // kernel function handle
    blocksPerGrid, 1, 1,      // Grid dimensions
    threadsPerBlock, 1, 1,    // Block dimensions
    0,                        // shared memory size
    0,                        // stream
    args,                     // kernel arguments
    nullptr                   // extra parameters
);
```

## Performance Analysis Tools

### 1. nvprof (Command Line Profiler)
```bash
# Basic profiling
nvprof ./your_program

# Detailed profiling with metrics
nvprof --metrics all ./your_program

# Timeline analysis
nvprof --print-gpu-trace ./your_program
```

### 2. Nsight Systems
- System-level performance analysis
- CPU-GPU timeline correlation
- Memory transfer analysis
- Kernel execution timeline

### 3. Nsight Compute
- Kernel-level performance analysis
- Instruction-level profiling
- Memory access pattern analysis
- Occupancy analysis

## Performance Optimization Strategies

### 1. Memory Access Optimization
- **Coalesced Access**: Ensure adjacent threads access adjacent memory addresses
- **Shared Memory Usage**: Cache frequently accessed data
- **Memory Alignment**: Align data to memory boundaries

### 2. Thread Block Optimization
- **Block Size**: Choose optimal thread block size (usually 256 or 512)
- **Grid Size**: Ensure sufficient blocks to utilize all SMs
- **Occupancy**: Maximize SM occupancy

### 3. Warp Divergence Avoidance
- **Branch Reduction**: Minimize conditional statements
- **Data Sorting**: Group similar data together
- **Algorithm Redesign**: Use branch-free algorithms

## Debugging Techniques

### 1. Error Handling
```cpp
CUresult result = cuLaunchKernel(kernel, ...);
if (result != CUDA_SUCCESS) {
    const char* errorString;
    cuGetErrorString(result, &errorString);
    std::cerr << "CUDA error: " << errorString << std::endl;
}
```

### 2. Memory Checking
```bash
# Use compute-sanitizer for memory error detection
nvcc -g -G -o debug_program debug_program.cu
compute-sanitizer ./debug_program
```

### 3. Boundary Validation
```cpp
// Always check array bounds
if (idx < n) {
    // Safe to access array[idx]
    result[idx] = input[idx];
}
```

## Quick Start

### 1. Compile PTX
```bash
nvcc -ptx -o vector_add.ptx vector_add.cu
```

### 2. Compile Driver API Program
```bash
nvcc -o run_ptx_manual run_ptx_manual.cu -lcuda
```

### 3. Run and Profile
```bash
# Basic execution
./run_ptx_manual

# Performance profiling
nvprof ./run_ptx_manual

# Memory checking
compute-sanitizer ./run_ptx_manual
```

## Summary

Today we have learned:
1. **CUDA Compilation Process**: Understanding PTX and CUBIN generation
2. **Driver API Usage**: Manual kernel loading and execution
3. **Performance Analysis**: Using profiling tools effectively
4. **Optimization Strategies**: Memory, thread, and algorithm optimization
5. **Debugging Techniques**: Error handling and memory validation

**Key Concepts**:
- **PTX**: Intermediate representation for cross-architecture portability
- **Driver API**: Low-level control for advanced applications
- **Performance Profiling**: Essential for optimization
- **Memory Optimization**: Critical for GPU performance

**Next Steps**:
- Practice with different optimization strategies
- Experiment with various thread block configurations
- Explore advanced profiling techniques

## 📁 Quick File Links

**Main Files**:
- [README.md](README.md) - This tutorial file
- [run_ptx_manual.cu](run_ptx_manual.cu) - PTX loading and execution example

**Generated Files**:
- [vector_add.ptx](vector_add.ptx) - PTX intermediate representation

**Compilation Commands**:
```bash
# Generate PTX
nvcc -ptx -o vector_add.ptx vector_add.cu

# Compile Driver API program
nvcc -o run_ptx_manual run_ptx_manual.cu -lcuda

# Run with profiling
nvprof ./run_ptx_manual
```
