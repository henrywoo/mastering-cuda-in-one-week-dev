# 🚀 CUDA编程一周速成课程 - 从入门到精通

## 概述

这是一个**7天速成**的CUDA编程完整课程，专为初级SDE设计，每天一个主题，循序渐进地教授CUDA编程的核心概念和实践技巧。通过一周的密集学习，你将掌握从基础概念到高级优化的完整知识体系，包括最新的LLM优化技术和不同GPU架构的优化策略。

**课程特色**：
- 📅 **7天完整课程**：每天一个主题，系统化学习
- 🎯 **循序渐进**：从基础概念到高级优化，层层递进
- 💻 **实战导向**：每个主题都有完整的代码示例和实践项目
- 🚀 **前沿技术**：涵盖最新的LLM优化技术和GPU架构特性

## 教程结构

每个Day的教程都包含完整的理论讲解、代码示例、实践项目和"相关文件快速链接"section，方便读者快速找到相关代码文件和学习资源。

**📚 完整教程链接**：
- [Day 1: CUDA编程基础 - 硬件架构与编程模型](day1/README.md)
- [Day 2: CUDA调试与优化 - PTX加载与性能分析](day2/README.md)
- [Day 3: 矩阵乘法优化 - CUDA性能调优实战](day3/README.md)
- [Day 4: 卷积神经网络(CNN) - CUDA深度学习实战](day4/README.md)
- [Day 5: 注意力机制和Transformer - 现代NLP的CUDA实现](day5/README.md)
- [Day 6: 最新LLM CUDA Kernel定制优化 - 前沿技术实战](day6/README.md)
- [Day 7: CUDA性能调优高级技巧 - 从理论到实践](day7/README.md)

### Day 1: CUDA编程基础 - 硬件架构与编程模型
- **学习目标**: 理解CUDA编程模型，掌握GPU硬件架构和内存层次结构
- **核心概念**: 线程层次结构、Warp执行特性、内存管理、动态kernel加载
- **实践项目**: 向量加法、向量点积、GPU配置工具、CUBIN文件运行
- **文件**: [day1/README.md](day1/README.md), [day1/vector_add.cu](day1/vector_add.cu), [day1/vector_dot.cu](day1/vector_dot.cu), [day1/run_cubin.cpp](day1/run_cubin.cpp), [day1/gpu_info.py](day1/gpu_info.py), [day1/GPU_CONFIG_SUMMARY.md](day1/GPU_CONFIG_SUMMARY.md)

### Day 2: CUDA调试与优化 - PTX加载与性能分析
- **学习目标**: 理解CUDA编译流程，掌握Driver API，学会调试和性能优化
- **核心概念**: PTX、CUBIN、CUDA上下文、性能分析工具、优化策略
- **实践项目**: 手动加载和执行PTX代码、性能分析和优化实战
- **文件**: [day2/README.md](day2/README.md), [day2/run_ptx_manual.cu](day2/run_ptx_manual.cu)

### Day 3: 矩阵乘法优化 - CUDA性能调优实战
- **学习目标**: 掌握矩阵乘法的CUDA实现和优化
- **核心概念**: 共享内存、内存合并访问、CUDA流
- **NVIDIA库**: cuBLAS、CUTLASS介绍和使用
- **实践项目**: 多种优化版本的矩阵乘法
- **文件**: [day3/README.md](day3/README.md), [day3/matrix_mul.cu](day3/matrix_mul.cu), [day3/matrix_mul_optimized.cu](day3/matrix_mul_optimized.cu), [day3/matrix_mul_cublas.cu](day3/matrix_mul_cublas.cu), [day3/matrix_mul_cutlass.cu](day3/matrix_mul_cutlass.cu)

### Day 4: 卷积神经网络(CNN) - CUDA深度学习实战
- **学习目标**: 实现CNN核心操作，理解卷积优化
- **核心概念**: 2D卷积、共享内存优化、分离卷积
- **NVIDIA库**: cuDNN库介绍和性能对比
- **实践项目**: 多种卷积算法的CUDA实现
- **文件**: [day4/README.md](day4/README.md), [day4/cnn_conv.cu](day4/cnn_conv.cu), [day4/cnn_conv_optimized.cu](day4/cnn_conv_optimized.cu), [day4/cnn_conv_cudnn.cu](day4/cnn_conv_cudnn.cu), [day4/cnn_forward.cu](day4/cnn_forward.cu)

### Day 5: 注意力机制和Transformer - 现代NLP的CUDA实现
- **学习目标**: 掌握注意力机制和Transformer架构
- **核心概念**: 自注意力、多头注意力、位置编码
- **实践项目**: 完整的Transformer实现
- **文件**: [day5/README.md](day5/README.md), [day5/self_attention.cu](day5/self_attention.cu), [day5/multi_head_attention.cu](day5/multi_head_attention.cu), [day5/transformer_block.cu](day5/transformer_block.cu), [day5/transformer.cu](day5/transformer.cu)

### Day 6: 最新LLM CUDA Kernel定制优化 - 前沿技术实战
- **学习目标**: 掌握最新的LLM优化技术
- **核心概念**: Flash Attention、Paged Attention、Grouped Query Attention
- **前沿技术**: 稀疏注意力、最新Tensor Core优化
- **实践项目**: 多种注意力优化算法的实现
- **文件**: [day6/README.md](day6/README.md), [day6/flash_attention.cu](day6/flash_attention.cu), [day6/paged_attention.cu](day6/paged_attention.cu), [day6/grouped_query_attention.cu](day6/grouped_query_attention.cu), [day6/sparse_attention.cu](day6/sparse_attention.cu), [day6/mixed_precision_attention.cu](day6/mixed_precision_attention.cu)

### Day 7: CUDA性能调优高级技巧 - 从理论到实践
- **学习目标**: 掌握高级性能调优技巧
- **核心概念**: 内存层次优化、指令级优化、架构特定优化
- **GPU架构**: 不同架构特性对比，包括最新的Blackwell架构
- **实践项目**: 性能分析和优化实战
- **文件**: [day7/README.md](day7/README.md), [day7/performance_tuning.cu](day7/performance_tuning.cu), [day7/blackwell_tuning.cu](day7/blackwell_tuning.cu), [day7/memory_optimization.cu](day7/memory_optimization.cu), [day7/instruction_optimization.cu](day7/instruction_optimization.cu)


## 🎯 一周学习总结

通过这**7天密集学习**，你将掌握：

- **Day 1**: CUDA编程基础 - 硬件架构与编程模型
- **Day 2**: CUDA调试与优化 - PTX加载与性能分析  
- **Day 3**: 矩阵乘法优化 - CUDA性能调优实战
- **Day 4**: 卷积神经网络(CNN) - CUDA深度学习实战
- **Day 5**: 注意力机制和Transformer - 现代NLP的CUDA实现
- **Day 6**: 最新LLM CUDA Kernel定制优化 - 前沿技术实战
- **Day 7**: CUDA性能调优高级技巧 - 从理论到实践

**一周学习成果**：
- 🎯 **理论基础**：完整的CUDA编程模型和硬件架构理解
- 💻 **实践技能**：从简单kernel到复杂深度学习模型的实现能力
- 🚀 **优化技巧**：掌握各种性能优化策略和工具使用
- 🔬 **前沿技术**：了解最新的LLM优化技术和GPU架构特性

---

**开始你的CUDA编程一周速成之旅吧！** 🚀

记住：实践是最好的老师，多写代码，多调试，多优化！

## 贡献和反馈

如果你发现教程中的错误或有改进建议，欢迎：

1. 提交Issue
2. 创建Pull Request
3. 发送邮件反馈

## 许可证

本教程采用MIT许可证，你可以自由使用、修改和分发。

