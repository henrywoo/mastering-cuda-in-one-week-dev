#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU信息获取程序 - Python版本
支持多种方式获取GPU信息，包括CUDA Python、nvidia-ml-py等
"""

import os
import sys
import subprocess
import platform
import json
from typing import Dict, List, Optional, Tuple

def run_command(cmd: str) -> Tuple[str, int]:
    """运行命令并返回输出和退出码"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "命令执行超时", -1
    except Exception as e:
        return f"命令执行错误: {e}", -1

def format_bytes(bytes_value: int) -> str:
    """格式化字节数为人类可读格式"""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit = 0
    size = float(bytes_value)
    
    while size >= 1024.0 and unit < len(units) - 1:
        size /= 1024.0
        unit += 1
    
    return f"{size:.2f} {units[unit]}"

def format_frequency(freq_khz: int) -> str:
    """格式化频率为GHz"""
    freq_ghz = freq_khz / 1000000.0
    return f"{freq_ghz:.2f} GHz"

def get_compute_capability_string(major: int, minor: int) -> str:
    """获取计算能力字符串"""
    return f"{major}.{minor}"

def get_architecture_name(major: int, minor: int) -> str:
    """获取GPU架构名称"""
    if major == 9:
        if minor == 0:
            return "Hopper"
        elif minor == 2:
            return "Blackwell"
    elif major == 8:
        if minor == 0:
            return "Ampere"
        elif minor == 6:
            return "Ada Lovelace"
        elif minor == 9:
            return "Hopper"
    elif major == 7:
        if minor == 0:
            return "Volta"
        elif minor == 2:
            return "Turing"
        elif minor == 5:
            return "Ampere"
    elif major == 6:
        return "Pascal"
    elif major == 5:
        return "Maxwell"
    elif major == 3:
        return "Kepler"
    elif major == 2:
        return "Fermi"
    elif major == 1:
        return "Tesla"
    return "Unknown"

def get_gpu_model_range(sm_count: int) -> str:
    """根据SM数量推测GPU型号"""
    if sm_count >= 132:
        return "H200/B200 (Blackwell)"
    elif sm_count >= 108:
        return "H100 (Hopper)"
    elif sm_count >= 84:
        return "A100 (Ampere)"
    elif sm_count >= 68:
        return "RTX 4090 (Ada Lovelace)"
    elif sm_count >= 56:
        return "RTX 4080/RTX 3090 (Ada/Ampere)"
    elif sm_count >= 46:
        return "RTX 3080/RTX 2080 Ti (Ampere/Turing)"
    elif sm_count >= 36:
        return "RTX 3070/RTX 2070 (Ampere/Turing)"
    elif sm_count >= 28:
        return "RTX 3060/RTX 2060 (Ampere/Turing)"
    elif sm_count >= 20:
        return "GTX 1660/GTX 1060 (Turing/Pascal)"
    elif sm_count >= 10:
        return "GTX 1050/GTX 950 (Pascal/Maxwell)"
    else:
        return "Unknown"

def get_system_info() -> Dict[str, str]:
    """获取系统信息"""
    info = {}
    
    # 操作系统信息
    info['os'] = platform.system()
    info['os_version'] = platform.version()
    info['architecture'] = platform.machine()
    info['processor'] = platform.processor()
    
    # Python信息
    info['python_version'] = sys.version
    
    # 尝试获取CPU信息
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    info['cpu_model'] = line.split(':')[1].strip()
                    break
    except:
        info['cpu_model'] = "无法获取"
    
    # 尝试获取内存信息
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    mem_kb = int(line.split()[1])
                    info['total_memory'] = format_bytes(mem_kb * 1024)
                    break
    except:
        info['total_memory'] = "无法获取"
    
    return info

def get_cuda_environment_info() -> Dict[str, str]:
    """获取CUDA环境信息"""
    info = {}
    
    # 环境变量
    info['CUDA_HOME'] = os.environ.get('CUDA_HOME', '未设置')
    info['CUDA_PATH'] = os.environ.get('CUDA_PATH', '未设置')
    info['PATH'] = '包含CUDA' if 'cuda' in os.environ.get('PATH', '').lower() else '不包含CUDA'
    
    # CUDA工具检查
    cuda_tools = [
        '/usr/local/cuda/bin/nvcc',
        '/usr/local/cuda/bin/nvidia-smi',
        '/usr/bin/nvcc',
        '/usr/bin/nvidia-smi'
    ]
    
    for tool in cuda_tools:
        info[f'tool_{os.path.basename(tool)}'] = '存在' if os.path.exists(tool) else '不存在'
    
    # 获取CUDA版本
    nvcc_output, nvcc_code = run_command('nvcc --version')
    if nvcc_code == 0:
        for line in nvcc_output.split('\n'):
            if 'release' in line.lower():
                info['cuda_version'] = line.strip()
                break
    else:
        info['cuda_version'] = '无法获取'
    
    return info

def get_gpu_info_nvidia_smi() -> List[Dict[str, str]]:
    """使用nvidia-smi获取GPU信息"""
    gpus = []
    
    # 基本信息
    output, code = run_command('nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits')
    if code == 0:
        for line in output.split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 6:
                    gpu_info = {
                        'name': parts[0],
                        'memory_total': f"{parts[1]} MB",
                        'memory_free': f"{parts[2]} MB",
                        'memory_used': f"{parts[3]} MB",
                        'utilization': f"{parts[4]}%",
                        'temperature': f"{parts[5]}°C",
                        'power': f"{parts[6]} W" if len(parts) > 6 else "N/A"
                    }
                    gpus.append(gpu_info)
    
    # 详细属性
    output, code = run_command('nvidia-smi -q')
    if code == 0:
        current_gpu = -1
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('GPU '):
                current_gpu += 1
            elif current_gpu >= 0 and current_gpu < len(gpus):
                if 'Product Name' in line:
                    gpus[current_gpu]['product_name'] = line.split(':')[1].strip()
                elif 'Driver Version' in line:
                    gpus[current_gpu]['driver_version'] = line.split(':')[1].strip()
                elif 'CUDA Version' in line:
                    gpus[current_gpu]['cuda_version'] = line.split(':')[1].strip()
                elif 'Compute Cap' in line:
                    gpus[current_gpu]['compute_capability'] = line.split(':')[1].strip()
    
    return gpus

def get_gpu_info_cuda_python() -> List[Dict[str, str]]:
    """使用CUDA Python获取GPU信息"""
    gpus = []
    
    try:
        import cuda
        import cuda.cuda as cuda_api
        
        # 获取设备数量
        device_count = cuda_api.cuDeviceGetCount()
        
        for device_id in range(device_count):
            gpu_info = {}
            
            # 获取设备属性
            device = cuda_api.cuDeviceGet(device_id)
            
            # 设备名称
            name = cuda_api.cuDeviceGetName(device)
            gpu_info['name'] = name.decode('utf-8')
            
            # 计算能力
            major = cuda_api.cuDeviceGetAttribute(cuda_api.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
            minor = cuda_api.cuDeviceGetAttribute(cuda_api.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
            gpu_info['compute_capability'] = f"{major}.{minor}"
            gpu_info['architecture'] = get_architecture_name(major, minor)
            
            # 多处理器数量
            sm_count = cuda_api.cuDeviceGetAttribute(cuda_api.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)
            gpu_info['sm_count'] = sm_count
            gpu_info['model_range'] = get_gpu_model_range(sm_count)
            
            # 内存信息
            total_mem = cuda_api.cuDeviceTotalMem(device)
            gpu_info['total_memory'] = format_bytes(total_mem)
            
            # 线程配置
            max_threads_per_block = cuda_api.cuDeviceGetAttribute(cuda_api.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device)
            gpu_info['max_threads_per_block'] = max_threads_per_block
            
            max_threads_per_sm = cuda_api.cuDeviceGetAttribute(cuda_api.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device)
            gpu_info['max_threads_per_sm'] = max_threads_per_sm
            
            # 共享内存
            shared_mem_per_block = cuda_api.cuDeviceGetAttribute(cuda_api.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, device)
            gpu_info['shared_mem_per_block'] = format_bytes(shared_mem_per_block)
            
            # 寄存器
            regs_per_block = cuda_api.cuDeviceGetAttribute(cuda_api.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device)
            gpu_info['regs_per_block'] = regs_per_block
            
            gpus.append(gpu_info)
            
    except ImportError:
        print("  CUDA Python未安装，跳过CUDA Python信息获取")
    except Exception as e:
        print(f"  CUDA Python信息获取失败: {e}")
    
    return gpus

def get_gpu_info_nvidia_ml() -> List[Dict[str, str]]:
    """使用nvidia-ml-py获取GPU信息"""
    gpus = []
    
    try:
        import pynvml
        
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for device_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
            gpu_info = {}
            
            # 基本信息
            name = pynvml.nvmlDeviceGetName(handle)
            gpu_info['name'] = name.decode('utf-8')
            
            # 内存信息
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info['total_memory'] = format_bytes(mem_info.total)
            gpu_info['free_memory'] = format_bytes(mem_info.free)
            gpu_info['used_memory'] = format_bytes(mem_info.used)
            
            # 温度
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            gpu_info['temperature'] = f"{temp}°C"
            
            # 功率
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为W
            gpu_info['power_usage'] = f"{power:.1f} W"
            
            # 利用率
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_info['gpu_utilization'] = f"{util.gpu}%"
            gpu_info['memory_utilization'] = f"{util.memory}%"
            
            gpus.append(gpu_info)
            
        pynvml.nvmlShutdown()
        
    except ImportError:
        print("  nvidia-ml-py未安装，跳过nvidia-ml信息获取")
    except Exception as e:
        print(f"  nvidia-ml信息获取失败: {e}")
    
    return gpus

def print_optimal_configuration(gpu_info: Dict[str, str]):
    """打印最佳配置建议"""
    print("【最佳配置建议】")
    
    # 线程块大小建议
    try:
        max_threads = int(gpu_info.get('max_threads_per_block', 1024))
        if max_threads >= 1024:
            optimal_threads = 512
        elif max_threads >= 512:
            optimal_threads = 256
        elif max_threads >= 256:
            optimal_threads = 128
        else:
            optimal_threads = max_threads
        
        print(f"  推荐线程块大小: {optimal_threads} 线程/块")
        print("    理由: 平衡了warp大小(32)、寄存器使用和线程切换开销")
        
        # 共享内存建议
        try:
            shared_mem = int(gpu_info.get('shared_mem_per_block', '0').split()[0])
            optimal_shared = shared_mem // 2
            print(f"  推荐共享内存使用: {optimal_shared} KB/块")
            print("    理由: 为其他资源留出空间，避免SM资源竞争")
        except:
            pass
        
        # 寄存器建议
        try:
            regs_per_block = int(gpu_info.get('regs_per_block', 0))
            optimal_regs_per_thread = regs_per_block // optimal_threads
            print(f"  推荐寄存器使用: {optimal_regs_per_thread} 寄存器/线程")
            print("    理由: 避免寄存器溢出到本地内存")
        except:
            pass
            
    except:
        print("  无法获取线程配置信息")
    
    print()

def print_performance_features(gpu_info: Dict[str, str]):
    """打印性能特征"""
    print("【性能特征】")
    
    try:
        sm_count = int(gpu_info.get('sm_count', 0))
        compute_cap = gpu_info.get('compute_capability', '0.0')
        major, minor = map(int, compute_cap.split('.'))
        
        # 估算核心数
        cores_per_sm = 0
        if major >= 8:
            cores_per_sm = 128  # Ampere及以后架构
        elif major >= 7:
            cores_per_sm = 64   # Volta/Turing架构
        elif major >= 6:
            cores_per_sm = 128  # Pascal架构
        else:
            cores_per_sm = 192  # 更早架构
        
        total_cores = sm_count * cores_per_sm
        print(f"  估算总核心数: {total_cores}")
        print(f"  架构: {gpu_info.get('architecture', 'Unknown')}")
        
    except:
        print("  无法获取性能特征信息")
    
    print()

def main():
    """主函数"""
    print("=== NVIDIA GPU 详细配置信息 (Python版本) ===\n")
    
    # 获取系统信息
    print("【系统信息】")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    print()
    
    # 获取CUDA环境信息
    print("【CUDA环境信息】")
    cuda_env = get_cuda_environment_info()
    for key, value in cuda_env.items():
        print(f"  {key}: {value}")
    print()
    
    # 尝试多种方式获取GPU信息
    gpu_info_list = []
    
    # 方法1: nvidia-smi
    print("【GPU信息获取 - nvidia-smi】")
    gpu_info_list = get_gpu_info_nvidia_smi()
    if gpu_info_list:
        for i, gpu in enumerate(gpu_info_list):
            print(f"GPU {i}:")
            for key, value in gpu.items():
                print(f"  {key}: {value}")
            print()
    else:
        print("  无法通过nvidia-smi获取GPU信息")
    print()
    
    # 方法2: CUDA Python
    print("【GPU信息获取 - CUDA Python】")
    cuda_gpu_info = get_gpu_info_cuda_python()
    if cuda_gpu_info:
        for i, gpu in enumerate(cuda_gpu_info):
            print(f"GPU {i}:")
            for key, value in gpu.items():
                print(f"  {key}: {value}")
            
            # 打印最佳配置建议
            print_optimal_configuration(gpu)
            print_performance_features(gpu)
    else:
        print("  无法通过CUDA Python获取GPU信息")
    print()
    
    # 方法3: nvidia-ml-py
    print("【GPU信息获取 - nvidia-ml-py】")
    ml_gpu_info = get_gpu_info_nvidia_ml()
    if ml_gpu_info:
        for i, gpu in enumerate(ml_gpu_info):
            print(f"GPU {i}:")
            for key, value in gpu.items():
                print(f"  {key}: {value}")
            print()
    else:
        print("  无法通过nvidia-ml-py获取GPU信息")
    print()
    
    # 性能优化建议
    print("【性能优化建议】")
    print("  1. 使用nvidia-smi监控GPU使用情况")
    print("  2. 使用nvprof或nsight compute进行性能分析")
    print("  3. 测试不同线程块大小找到最优配置")
    print("  4. 监控共享内存和寄存器使用情况")
    print("  5. 使用cuda-memcheck检查内存错误")
    print("  6. 考虑使用CUDA Occupancy Calculator")
    print("  7. 安装CUDA Python: pip install cuda-python")
    print("  8. 安装nvidia-ml-py: pip install nvidia-ml-py3")
    print()
    
    print("=== 程序执行完成 ===")

if __name__ == "__main__":
    main()
