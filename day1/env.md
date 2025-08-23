# CUDA开发环境搭建指南

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GeForce 8系列及以上，或Tesla/Quadro系列
- **内存**: 建议8GB以上
- **存储**: 至少10GB可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+, CentOS 7+, RHEL 7+)
- **编译器**: GCC 7.0+ 或 Clang 5.0+
- **内核**: Linux 3.10+ (推荐4.15+)

## Ubuntu/Debian系统安装

### 方法1：使用软件包管理器（推荐新手）

```bash
# 更新软件包列表
sudo apt update

# 安装CUDA工具包（包含nvcc编译器）
sudo apt install nvidia-cuda-toolkit

# 验证安装
nvcc --version
```

### 方法2：从NVIDIA官网下载安装（推荐开发者）

1. 访问 [NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-downloads)
2. 选择你的操作系统和版本
3. 按照官方指导进行安装

## CentOS/RHEL系统安装

```bash
# 安装EPEL仓库
sudo yum install epel-release

# 安装CUDA工具包
sudo yum install cuda

# 验证安装
nvcc --version
```

## 验证安装

### 检查GPU驱动
```bash
# 检查NVIDIA驱动是否正常
nvidia-smi

# 应该看到类似输出：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.4     |
# +-----------------------------------------------------------------------------+
```

### 检查CUDA编译器
```bash
# 检查CUDA编译器版本
nvcc --version

# 应该看到类似输出：
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2021 NVIDIA Corporation
# Built on Sun_Aug_15_21:14:11_PDT_2021
# Cuda compilation tools, release 11.4, V11.4.120
```

### 检查CUDA运行时
```bash
# 检查CUDA运行时库
nvcc -V

# 检查CUDA示例程序
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

## 环境变量配置

### 自动配置（推荐）
```bash
# 将以下内容添加到 ~/.bashrc 或 ~/.zshrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# 重新加载配置
source ~/.bashrc
```

### 手动配置
```bash
# 创建CUDA环境配置文件
sudo tee /etc/profile.d/cuda.sh << EOF
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
EOF

# 重新登录或执行
source /etc/profile.d/cuda.sh
```

## 常见问题解决

### 权限问题
```bash
# 如果遇到权限问题，将用户添加到video组
sudo usermod -a -G video $USER

# 重新登录后生效
```

### 驱动版本不匹配
```bash
# 检查驱动和CUDA版本兼容性
nvidia-smi
nvcc --version

# 如果版本不匹配，需要重新安装匹配的驱动
```

### 编译错误
```bash
# 检查CUDA路径
echo $CUDA_HOME
which nvcc

# 检查库文件
ls -la /usr/local/cuda/lib64/
```

## 开发工具推荐

### IDE和编辑器
- **Visual Studio Code**: 安装CUDA扩展
- **CLion**: 支持CUDA项目
- **Eclipse**: 免费CUDA开发环境

### 调试工具
- **cuda-gdb**: CUDA调试器
- **compute-sanitizer**: 内存检查工具
- **nvprof**: 性能分析工具

### 示例代码
```bash
# 编译第一个CUDA程序
nvcc -o hello_cuda hello_cuda.cu

# 运行程序
./hello_cuda
```

## 版本兼容性

### CUDA版本与驱动版本对应关系
| CUDA版本 | 最低驱动版本 | 推荐驱动版本 |
|----------|-------------|-------------|
| CUDA 12.x | 525.60.13+ | 535.54.03+ |
| CUDA 11.8 | 450.80.02+ | 525.60.13+ |
| CUDA 11.6 | 450.80.02+ | 510.47.03+ |
| CUDA 11.4 | 450.80.02+ | 470.82.01+ |

### 操作系统兼容性
- **Ubuntu 22.04**: 支持CUDA 11.0+
- **Ubuntu 20.04**: 支持CUDA 10.2+
- **CentOS 8**: 支持CUDA 10.2+
- **CentOS 7**: 支持CUDA 9.0+

## 参考资源

- [NVIDIA CUDA官方文档](https://docs.nvidia.com/cuda/)
- [CUDA安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
