#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

// 格式化字节数为人类可读格式
std::string formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

// 格式化频率为GHz
std::string formatFrequency(int freqKHz) {
    double freqGHz = freqKHz / 1000000.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << freqGHz << " GHz";
    return oss.str();
}

// 获取计算能力字符串
std::string getComputeCapabilityString(int major, int minor) {
    std::ostringstream oss;
    oss << major << "." << minor;
    return oss.str();
}

// 获取GPU架构名称
std::string getArchitectureName(int major, int minor) {
    if (major == 9) {
        if (minor == 0) return "Hopper";
        else if (minor == 2) return "Blackwell";
    } else if (major == 8) {
        if (minor == 0) return "Ampere";
        else if (minor == 6) return "Ada Lovelace";
        else if (minor == 9) return "Hopper";
    } else if (major == 7) {
        if (minor == 0) return "Volta";
        else if (minor == 2) return "Turing";
        else if (minor == 5) return "Ampere";
    } else if (major == 6) {
        if (minor == 0) return "Pascal";
        else if (minor == 1) return "Pascal";
    } else if (major == 5) {
        return "Maxwell";
    } else if (major == 3) {
        return "Kepler";
    } else if (major == 2) {
        return "Fermi";
    } else if (major == 1) {
        return "Tesla";
    }
    return "Unknown";
}

// 获取SM数量对应的GPU型号范围
std::string getGPUModelRange(int smCount) {
    if (smCount >= 132) return "H200/B200 (Blackwell)";
    else if (smCount >= 108) return "H100 (Hopper)";
    else if (smCount >= 84) return "A100 (Ampere)";
    else if (smCount >= 68) return "RTX 4090 (Ada Lovelace)";
    else if (smCount >= 56) return "RTX 4080/RTX 3090 (Ada/Ampere)";
    else if (smCount >= 46) return "RTX 3080/RTX 2080 Ti (Ampere/Turing)";
    else if (smCount >= 36) return "RTX 3070/RTX 2070 (Ampere/Turing)";
    else if (smCount >= 28) return "RTX 3060/RTX 2060 (Ampere/Turing)";
    else if (smCount >= 20) return "GTX 1660/GTX 1060 (Turing/Pascal)";
    else if (smCount >= 10) return "GTX 1050/GTX 950 (Pascal/Maxwell)";
    else return "Unknown";
}

int main() {
    std::cout << "=== NVIDIA GPU 详细配置信息 ===\n\n";
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "错误: 未找到支持CUDA的GPU设备\n";
        return -1;
    }
    
    std::cout << "检测到 " << deviceCount << " 个CUDA设备\n\n";
    
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        std::cout << "=== 设备 " << device << ": " << prop.name << " ===\n";
        std::cout << std::string(50, '-') << "\n\n";
        
        // 基本信息
        std::cout << "【基本信息】\n";
        std::cout << "  设备名称: " << prop.name << "\n";
        std::cout << "  计算能力: " << getComputeCapabilityString(prop.major, prop.minor) << "\n";
        std::cout << "  架构名称: " << getArchitectureName(prop.major, prop.minor) << "\n";
        std::cout << "  多处理器数量: " << prop.multiProcessorCount << " (SM)\n";
        std::cout << "  推测型号: " << getGPUModelRange(prop.multiProcessorCount) << "\n\n";
        
        // 时钟频率
        std::cout << "【时钟频率】\n";
        std::cout << "  核心时钟: " << formatFrequency(prop.clockRate) << "\n";
        std::cout << "  内存时钟: " << formatFrequency(prop.memoryClockRate) << "\n";
        std::cout << "  内存总线宽度: " << prop.memoryBusWidth << " 位\n\n";
        
        // 内存配置
        std::cout << "【内存配置】\n";
        std::cout << "  全局内存: " << formatBytes(prop.totalGlobalMem) << "\n";
        std::cout << "  常量内存: " << formatBytes(prop.totalConstMem) << "\n";
        std::cout << "  共享内存/块: " << formatBytes(prop.sharedMemPerBlock) << "\n";
        std::cout << "  寄存器/块: " << prop.regsPerBlock << " 个\n";
        std::cout << "  L2缓存: " << formatBytes(prop.l2CacheSize) << "\n\n";
        
        // 线程配置
        std::cout << "【线程配置】\n";
        std::cout << "  每块最大线程数: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  每SM最大线程数: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  每SM最大块数: " << prop.maxBlocksPerMultiProcessor << "\n";
        std::cout << "  每块最大共享内存: " << formatBytes(prop.sharedMemPerBlock) << "\n";
        std::cout << "  每块最大寄存器: " << prop.regsPerBlock << "\n\n";
        
        // 线程块维度限制
        std::cout << "【线程块维度限制】\n";
        std::cout << "  最大线程块尺寸: (" << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
        std::cout << "  最大网格尺寸: (" << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n\n";
        
        // 最佳配置建议
        std::cout << "【最佳配置建议】\n";
        
        // 线程块大小建议
        int optimalThreadsPerBlock = 256;
        if (prop.maxThreadsPerBlock >= 1024) {
            optimalThreadsPerBlock = 512;
        } else if (prop.maxThreadsPerBlock >= 512) {
            optimalThreadsPerBlock = 256;
        } else if (prop.maxThreadsPerBlock >= 256) {
            optimalThreadsPerBlock = 128;
        } else {
            optimalThreadsPerBlock = prop.maxThreadsPerBlock;
        }
        
        std::cout << "  推荐线程块大小: " << optimalThreadsPerBlock << " 线程/块\n";
        std::cout << "    理由: 平衡了warp大小(32)、寄存器使用和线程切换开销\n";
        
        // 共享内存使用建议
        size_t optimalSharedMem = prop.sharedMemPerBlock / 2;  // 使用一半避免过度分配
        std::cout << "  推荐共享内存使用: " << formatBytes(optimalSharedMem) << "/块\n";
        std::cout << "    理由: 为其他资源留出空间，避免SM资源竞争\n";
        
        // 寄存器使用建议
        int optimalRegsPerThread = prop.regsPerBlock / optimalThreadsPerBlock;
        std::cout << "  推荐寄存器使用: " << optimalRegsPerThread << " 寄存器/线程\n";
        std::cout << "    理由: 避免寄存器溢出到本地内存\n\n";
        
        // 性能特征
        std::cout << "【性能特征】\n";
        
        // 理论峰值计算
        int coresPerSM = 0;
        if (prop.major >= 8) {
            coresPerSM = 128;  // Ampere及以后架构
        } else if (prop.major >= 7) {
            coresPerSM = 64;   // Volta/Turing架构
        } else if (prop.major >= 6) {
            coresPerSM = 128;  // Pascal架构
        } else {
            coresPerSM = 192;  // 更早架构
        }
        
        int totalCores = prop.multiProcessorCount * coresPerSM;
        double peakFreq = prop.clockRate / 1000000.0;  // GHz
        double theoreticalPeak = totalCores * peakFreq * 2;  // 2 FLOPS per core per cycle
        
        std::cout << "  理论峰值性能: " << std::fixed << std::setprecision(1) 
                  << theoreticalPeak << " GFLOPS (FP32)\n";
        std::cout << "  总核心数: " << totalCores << " (估算)\n";
        std::cout << "  核心频率: " << std::fixed << std::setprecision(2) << peakFreq << " GHz\n";
        
        // 内存带宽
        double memoryBandwidth = (prop.memoryClockRate * 2 * prop.memoryBusWidth) / 8.0 / 1e6;  // GB/s
        std::cout << "  理论内存带宽: " << std::fixed << std::setprecision(0) << memoryBandwidth << " GB/s\n\n";
        
        // 特殊功能支持
        std::cout << "【特殊功能支持】\n";
        std::cout << "  统一内存: " << (prop.unifiedAddressing ? "支持" : "不支持") << "\n";
        std::cout << "  异步内存操作: " << (prop.asyncEngineCount > 0 ? "支持" : "不支持") << "\n";
        std::cout << "  并发kernel: " << (prop.concurrentKernels ? "支持" : "不支持") << "\n";
        std::cout << "  流式多处理器: " << (prop.streamPrioritiesSupported ? "支持" : "不支持") << "\n";
        std::cout << "  全局L1缓存: " << (prop.globalL1CacheSupported ? "支持" : "不支持") << "\n";
        std::cout << "  本地L1缓存: " << (prop.localL1CacheSupported ? "支持" : "不支持") << "\n";
        
        // Tensor Core支持
        if (prop.major >= 7) {
            std::cout << "  Tensor Core: 支持 (FP16/BF16)\n";
        } else {
            std::cout << "  Tensor Core: 不支持\n";
        }
        
        // RT Core支持 (光线追踪)
        if (prop.major >= 7 && prop.minor >= 5) {
            std::cout << "  RT Core: 支持 (光线追踪)\n";
        } else {
            std::cout << "  RT Core: 不支持\n";
        }
        
        std::cout << "\n";
        
        // 实际测试建议
        std::cout << "【实际测试建议】\n";
        std::cout << "  1. 使用nvprof或nsight compute进行性能分析\n";
        std::cout << "  2. 测试不同线程块大小(64, 128, 256, 512)找到最优值\n";
        std::cout << "  3. 监控共享内存和寄存器使用情况\n";
        std::cout << "  4. 使用cuda-memcheck检查内存错误\n";
        std::cout << "  5. 考虑使用CUDA Occupancy Calculator优化配置\n\n";
        
        if (device < deviceCount - 1) {
            std::cout << std::string(80, '=') << "\n\n";
        }
    }
    
    std::cout << "=== 配置信息获取完成 ===\n";
    return 0;
}
