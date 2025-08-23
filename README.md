# 🚀 CUDA Programming One-Week Crash Course - From Beginner to Expert

## Overview

This is a **7-day crash course** in CUDA programming, designed specifically for junior SDEs. Each day covers one topic, progressively teaching the core concepts and practical techniques of CUDA programming. Through one week of intensive learning, you will master the complete knowledge system from basic concepts to advanced optimization, including the latest LLM optimization techniques and optimization strategies for different GPU architectures.

**Course Features**:
- 📅 **7-Day Complete Course**: One topic per day, systematic learning
- 🎯 **Progressive Learning**: From basic concepts to advanced optimization, step by step
- 💻 **Practice-Oriented**: Each topic has complete code examples and practical projects
- 🚀 **Cutting-Edge Technology**: Covers the latest LLM optimization techniques and GPU architecture features

## Course Structure

Each Day's tutorial contains complete theoretical explanations, code examples, practical projects, and a "Quick File Links" section, making it easy for readers to quickly find relevant code files and learning resources.

**📚 Complete Tutorial Links**:
- [Day 1: CUDA Programming Basics - Hardware Architecture and Programming Model](day1/README.md)
- [Day 2: CUDA Debugging and Optimization - PTX Loading and Performance Analysis](day2/README.md)
- [Day 3: Matrix Multiplication Optimization - CUDA Performance Tuning Practice](day3/README.md)
- [Day 4: Convolutional Neural Networks (CNN) - CUDA Deep Learning Practice](day4/README.md)
- [Day 5: Attention Mechanism and Transformer - Modern NLP CUDA Implementation](day5/README.md)
- [Day 6: Latest LLM CUDA Kernel Customization and Optimization - Cutting-Edge Technology Practice](day6/README.md)
- [Day 7: Advanced CUDA Performance Tuning Techniques - From Theory to Practice](day7/README.md)

### Day 1: CUDA Programming Basics - Hardware Architecture and Programming Model
- **Learning Objectives**: Understand CUDA programming model, master GPU hardware architecture and memory hierarchy
- **Core Concepts**: Thread hierarchy, Warp execution characteristics, memory management, dynamic kernel loading
- **Practical Projects**: Vector addition, vector dot product, GPU configuration tools, CUBIN file execution
- **Files**: [day1/README.md](day1/README.md), [day1/vector_add.cu](day1/vector_add.cu), [day1/vector_dot.cu](day1/vector_dot.cu), [day1/run_cubin.cpp](day1/run_cubin.cpp), [day1/gpu_info.py](day1/gpu_info.py), [day1/GPU_CONFIG_SUMMARY.md](day1/GPU_CONFIG_SUMMARY.md)

### Day 2: CUDA Debugging and Optimization - PTX Loading and Performance Analysis
- **Learning Objectives**: Understand CUDA compilation process, master Driver API, learn debugging and performance optimization
- **Core Concepts**: PTX, CUBIN, CUDA context, performance analysis tools, optimization strategies
- **Practical Projects**: Manual loading and execution of PTX code, performance analysis and optimization practice
- **Files**: [day2/README.md](day2/README.md), [day2/run_ptx_manual.cu](day2/run_ptx_manual.cu)

### Day 3: Matrix Multiplication Optimization - CUDA Performance Tuning Practice
- **Learning Objectives**: Master CUDA implementation and optimization of matrix multiplication
- **Core Concepts**: Shared memory, memory coalescing access, CUDA streams
- **NVIDIA Libraries**: Introduction and usage of cuBLAS, CUTLASS
- **Practical Projects**: Multiple optimized versions of matrix multiplication
- **Files**: [day3/README.md](day3/README.md), [day3/matrix_mul.cu](day3/matrix_mul.cu), [day3/matrix_mul_optimized.cu](day3/matrix_mul_optimized.cu), [day3/matrix_mul_cublas.cu](day3/matrix_mul_cublas.cu), [day3/matrix_mul_cutlass.cu](day3/matrix_mul_cutlass.cu)

### Day 4: Convolutional Neural Networks (CNN) - CUDA Deep Learning Practice
- **Learning Objectives**: Implement CNN core operations, understand convolution optimization
- **Core Concepts**: 2D convolution, shared memory optimization, separable convolution
- **NVIDIA Libraries**: Introduction and performance comparison of cuDNN library
- **Practical Projects**: Multiple convolution algorithm CUDA implementations
- **Files**: [day4/README.md](day4/README.md), [day4/cnn_conv.cu](day4/cnn_conv.cu), [day4/cnn_conv_optimized.cu](day4/cnn_conv_optimized.cu), [day4/cnn_conv_cudnn.cu](day4/cnn_conv_cudnn.cu), [day4/cnn_forward.cu](day4/cnn_forward.cu)

### Day 5: Attention Mechanism and Transformer - Modern NLP CUDA Implementation
- **Learning Objectives**: Master attention mechanism and Transformer architecture
- **Core Concepts**: Self-attention, multi-head attention, positional encoding
- **Practical Projects**: Complete Transformer implementation
- **Files**: [day5/README.md](day5/README.md), [day5/self_attention.cu](day5/self_attention.cu), [day5/multi_head_attention.cu](day5/multi_head_attention.cu), [day5/transformer_block.cu](day5/transformer_block.cu), [day5/transformer.cu](day5/transformer.cu)

### Day 6: Latest LLM CUDA Kernel Customization and Optimization - Cutting-Edge Technology Practice
- **Learning Objectives**: Master the latest LLM optimization techniques
- **Core Concepts**: Flash Attention, Paged Attention, Grouped Query Attention
- **Cutting-Edge Technology**: Sparse attention, latest Tensor Core optimization
- **Practical Projects**: Implementation of multiple attention optimization algorithms
- **Files**: [day6/README.md](day6/README.md), [day6/flash_attention.cu](day6/flash_attention.cu), [day6/paged_attention.cu](day6/paged_attention.cu), [day6/grouped_query_attention.cu](day6/grouped_query_attention.cu), [day6/sparse_attention.cu](day6/sparse_attention.cu), [day6/mixed_precision_attention.cu](day6/mixed_precision_attention.cu)

### Day 7: Advanced CUDA Performance Tuning Techniques - From Theory to Practice
- **Learning Objectives**: Master advanced performance tuning techniques
- **Core Concepts**: Memory hierarchy optimization, instruction-level optimization, architecture-specific optimization
- **GPU Architecture**: Comparison of different architecture features, including the latest Blackwell architecture
- **Practical Projects**: Performance analysis and optimization practice
- **Files**: [day7/README.md](day7/README.md), [day7/performance_tuning.cu](day7/performance_tuning.cu), [day7/blackwell_tuning.cu](day7/blackwell_tuning.cu), [day7/memory_optimization.cu](day7/memory_optimization.cu), [day7/instruction_optimization.cu](day7/instruction_optimization.cu)

## 🎯 One-Week Learning Summary

**Day 1**: Master CUDA programming model, understand GPU hardware architecture and memory hierarchy
**Day 2**: Learn debugging techniques and performance optimization fundamentals
**Day 3**: Implement and optimize matrix multiplication algorithms
**Day 4**: Build CNN models with CUDA acceleration
**Day 5**: Implement Transformer architecture and attention mechanisms
**Day 6**: Master cutting-edge LLM optimization techniques
**Day 7**: Advanced performance tuning and architecture-specific optimization

Through this intensive one-week course, you will transform from a CUDA beginner to an expert capable of implementing and optimizing complex deep learning models!

Remember: Practice is the best teacher - write more code, debug more, optimize more!

## Contributing and Feedback

If you find errors in the tutorial or have improvement suggestions, welcome to:

1. Submit an Issue
2. Create a Pull Request
3. Send email feedback

## License

This tutorial is licensed under MIT License. You are free to use, modify, and distribute it.

