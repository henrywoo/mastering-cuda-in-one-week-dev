# CUDA Development Environment Setup Guide

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GeForce 8 series and above, or Tesla/Quadro series
- **Memory**: 8GB+ recommended
- **Storage**: At least 10GB available space

### Software Requirements
- **Operating System**: Linux (Ubuntu 18.04+, CentOS 7+, RHEL 7+)
- **Compiler**: GCC 7.0+ or Clang 5.0+
- **Kernel**: Linux 3.10+ (4.15+ recommended)

## Ubuntu/Debian System Installation

### Method 1: Using Package Manager (Recommended for Beginners)

```bash
# Update package list
sudo apt update

# Install CUDA toolkit (includes nvcc compiler)
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

### Method 2: Download from NVIDIA Official Website (Recommended for Developers)

1. Visit [NVIDIA CUDA Download Page](https://developer.nvidia.com/cuda-downloads)
2. Select your operating system and version
3. Follow the official installation guide

## CentOS/RHEL System Installation

```bash
# Install EPEL repository
sudo yum install epel-release

# Install CUDA toolkit
sudo yum install cuda

# Verify installation
nvcc --version
```

## Verification

### Check GPU Driver
```bash
# Check if NVIDIA driver is working properly
nvidia-smi

# Should see output like:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.4     |
# +-----------------------------------------------------------------------------+
```

### Check CUDA Compiler
```bash
# Check CUDA compiler version
nvcc --version

# Should see output like:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2021 NVIDIA Corporation
# Built on Sun_Aug_15_21:14:11_PDT_2021
# Cuda compilation tools, release 11.4, V11.4.120
```

### Check CUDA Runtime
```bash
# Check CUDA runtime library
nvcc -V

# Check CUDA sample programs
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

## Environment Variable Configuration

### Automatic Configuration (Recommended)
```bash
# Add the following to ~/.bashrc or ~/.zshrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Reload configuration
source ~/.bashrc
```

### Manual Configuration
```bash
# Create CUDA environment configuration file
sudo tee /etc/profile.d/cuda.sh << EOF
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
EOF

# Relogin or execute
source /etc/profile.d/cuda.sh
```

## Common Issues and Solutions

### Permission Issues
```bash
# If you encounter permission issues, add user to video group
sudo usermod -a -G video $USER

# Takes effect after relogin
```

### Driver Version Mismatch
```bash
# Check driver and CUDA version compatibility
nvidia-smi
nvcc --version

# If versions don't match, reinstall matching driver
```

### Compilation Errors
```bash
# Check CUDA path
echo $CUDA_HOME
which nvcc

# Check library files
ls -la /usr/local/cuda/lib64/
```

## Development Tools Recommendations

### IDEs and Editors
- **Visual Studio Code**: Install CUDA extension
- **CLion**: Supports CUDA projects
- **Eclipse**: Free CUDA development environment

### Debugging Tools
- **cuda-gdb**: CUDA debugger
- **compute-sanitizer**: Memory checking tool
- **nvprof**: Performance analysis tool

### Sample Code
```bash
# Compile first CUDA program
nvcc -o hello_cuda hello_cuda.cu

# Run program
./hello_cuda
```

## Version Compatibility

### CUDA Version vs Driver Version Compatibility
| CUDA Version | Minimum Driver Version | Recommended Driver Version |
|--------------|------------------------|----------------------------|
| CUDA 12.x    | 525.60.13+            | 535.54.03+                |
| CUDA 11.8    | 450.80.02+            | 525.60.13+                |
| CUDA 11.6    | 450.80.02+            | 510.47.03+                |
| CUDA 11.4    | 450.80.02+            | 470.82.01+                |

### Operating System Compatibility
- **Ubuntu 22.04**: Supports CUDA 11.0+
- **Ubuntu 20.04**: Supports CUDA 10.2+
- **CentOS 8**: Supports CUDA 10.2+
- **CentOS 7**: Supports CUDA 9.0+

## Reference Resources

- [NVIDIA CUDA Official Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
