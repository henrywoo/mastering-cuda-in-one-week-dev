#!/bin/bash

echo "=== 编译所有Day 6 CUDA程序 ==="
echo ""

# 设置CUDA路径
CUDA_PATH="/usr/local/cuda-12.4"
NVCC="$CUDA_PATH/bin/nvcc"
INCLUDE_FLAGS="-I$CUDA_PATH/include"
LIB_FLAGS="-L$CUDA_PATH/lib64"

# 编译Flash Attention
echo "编译 Flash Attention..."
$NVCC $INCLUDE_FLAGS $LIB_FLAGS -O3 -arch=sm_70 -o flash_attention flash_attention.cu
if [ $? -eq 0 ]; then
    echo "✓ Flash Attention 编译成功"
else
    echo "✗ Flash Attention 编译失败"
fi
echo ""

# 编译Grouped Query Attention
echo "编译 Grouped Query Attention..."
$NVCC $INCLUDE_FLAGS $LIB_FLAGS -O3 -arch=sm_70 -o grouped_query_attention grouped_query_attention.cu
if [ $? -eq 0 ]; then
    echo "✓ Grouped Query Attention 编译成功"
else
    echo "✗ Grouped Query Attention 编译失败"
fi
echo ""

# 编译Mixed Precision Attention
echo "编译 Mixed Precision Attention..."
$NVCC $INCLUDE_FLAGS $LIB_FLAGS -O3 -arch=sm_70 -o mixed_precision_attention mixed_precision_attention.cu
if [ $? -eq 0 ]; then
    echo "✓ Mixed Precision Attention 编译成功"
else
    echo "✗ Mixed Precision Attention 编译失败"
fi
echo ""

# 编译Sparse Attention
echo "编译 Sparse Attention..."
$NVCC $INCLUDE_FLAGS $LIB_FLAGS -O3 -arch=sm_70 -o sparse_attention sparse_attention.cu
if [ $? -eq 0 ]; then
    echo "✓ Sparse Attention 编译成功"
else
    echo "✗ Sparse Attention 编译失败"
fi
echo ""

echo "=== 编译完成 ==="
echo ""
echo "生成的可执行文件:"
ls -la flash_attention grouped_query_attention mixed_precision_attention sparse_attention 2>/dev/null | grep -E "^-|^l"
echo ""
echo "运行示例:"
echo "  ./flash_attention"
echo "  ./grouped_query_attention"
echo "  ./mixed_precision_attention"
echo "  ./sparse_attention"
