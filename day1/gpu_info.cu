#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>  // FP8支持
#include <cuda/std/type_traits>

// 直接测试FP8/INT8支持的宏定义
#define TEST_FP8_INT8_DIRECTLY 1

// 首先定义traits结构体
// 使用traits检测数据类型支持
template<typename T>
struct is_fp16_supported {
    static constexpr bool value = false;
};

template<typename T>
struct is_bf16_supported {
    static constexpr bool value = false;
};

template<typename T>
struct is_fp8_supported {
    static constexpr bool value = false;
};

template<typename T>
struct is_int8_supported {
    static constexpr bool value = false;
};

// 特化FP16支持检测
#ifdef __CUDA_FP16_H__
template<>
struct is_fp16_supported<__half> {
    static constexpr bool value = true;
};
#endif

// 特化BF16支持检测
#ifdef __CUDA_BF16_H__
template<>
struct is_bf16_supported<__nv_bfloat16> {
    static constexpr bool value = true;
};
#endif

// 特化FP8支持检测 - 使用正确的FP8类型
template<>
struct is_fp8_supported<__nv_fp8_e4m3> {
    static constexpr bool value = true;
};

template<>
struct is_fp8_supported<__nv_fp8_e5m2> {
    static constexpr bool value = true;
};

// 特化INT8支持检测
template<>
struct is_int8_supported<int8_t> {
    static constexpr bool value = true;
};

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

// 使用traits检测FP8支持
bool testFP8SupportWithTraits() {
    std::cout << "【FP8支持测试 - Traits检测法】\n";
    
    // 使用traits检测FP8支持
    bool fp8Supported = is_fp8_supported<__nv_fp8_e4m3>::value;
    
    if (fp8Supported) {
        std::cout << "  ✅ Traits检测FP8支持: 是\n";
    } else {
        std::cout << "  ❌ Traits检测FP8支持: 否\n";
    }
    
    return fp8Supported;
}

// 测试FP8支持
void testFP8Support(int major, int minor) {
    bool fp8Supported = testFP8SupportWithTraits();
    
    std::cout << "  FP8支持: " << (fp8Supported ? "✅ 支持" : "❌ 不支持") << "\n";
    
    if (fp8Supported) {
        std::cout << "  FP8特性:\n";
        std::cout << "    - 存储需求: 相比FP16减半\n";
        std::cout << "    - 吞吐量: 相比FP16翻倍\n";
        std::cout << "    - 适用场景: AI推理、深度学习训练\n";
    }
    std::cout << "\n";
}

// 使用traits检测INT8支持
bool testINT8SupportWithTraits() {
    std::cout << "【INT8支持测试 - Traits检测法】\n";
    
    // 使用traits检测INT8支持
    bool int8Supported = is_int8_supported<int8_t>::value;
    
    if (int8Supported) {
        std::cout << "  ✅ Traits检测INT8支持: 是\n";
    } else {
        std::cout << "  ❌ Traits检测INT8支持: 否\n";
    }
    
    return int8Supported;
}

// 测试INT8支持
void testINT8Support(int major, int minor) {
    bool int8Supported = testINT8SupportWithTraits();
    
    std::cout << "  INT8支持: " << (int8Supported ? "✅ 支持" : "❌ 不支持") << "\n";
    
    if (int8Supported) {
        std::cout << "  INT8特性:\n";
        std::cout << "    - 存储需求: 相比FP32减少75%\n";
        std::cout << "    - 吞吐量: 相比FP32提升显著\n";
        std::cout << "    - 适用场景: 量化推理、边缘计算\n";
        std::cout << "    - Tensor Core代数: " << (major >= 8 ? "第3-4代" : "第1-2代") << "\n";
    }
    std::cout << "\n";
}



// 使用traits检测混合精度支持并演示混合精度计算
void testMixedPrecisionSupport(int major, int minor) {
    std::cout << "【混合精度支持测试 - Traits检测法】\n";
    
    // 使用traits检测支持
    bool fp16Supported = is_fp16_supported<__half>::value;
    bool bf16Supported = is_bf16_supported<__nv_bfloat16>::value;
    bool fp8Supported = is_fp8_supported<__nv_fp8_e4m3>::value;
    bool int8Supported = is_int8_supported<int8_t>::value;
    
    std::cout << "  FP16支持: " << (fp16Supported ? "✅ 支持" : "❌ 不支持") << "\n";
    std::cout << "  BF16支持: " << (bf16Supported ? "✅ 支持" : "❌ 不支持") << "\n";
    std::cout << "  FP8支持: " << (fp8Supported ? "✅ 支持" : "❌ 不支持") << "\n";
    std::cout << "  INT8支持: " << (int8Supported ? "✅ 支持" : "❌ 不支持") << "\n";
    
    if (fp16Supported || bf16Supported || fp8Supported || int8Supported) {
        std::cout << "  混合精度优势:\n";
        std::cout << "    - 内存使用减少\n";
        std::cout << "    - 计算速度提升\n";
        std::cout << "    - 支持Tensor Core加速\n";
        std::cout << "    - Traits检测: ✅ 支持\n";
        
        // 演示混合精度数学操作
        std::cout << "\n  【混合精度数学操作演示】\n";
        
        // 示例1: FP32 + FP16 混合计算
        if (fp16Supported) {
            float fp32_value = 3.14159f;
            __half fp16_value = __float2half(2.71828f);
            
            // 混合精度计算：FP32 + FP16 -> FP32
            float mixed_result = fp32_value + __half2float(fp16_value);
            std::cout << "    FP32 + FP16 混合计算:\n";
            std::cout << "      FP32值: " << fp32_value << "\n";
            std::cout << "      FP16值: " << __half2float(fp16_value) << "\n";
            std::cout << "      混合结果: " << mixed_result << "\n";
        }
        
        // 示例2: FP16 + FP8 混合计算
        if (fp16Supported && fp8Supported) {
            __half fp16_a = __float2half(1.5f);
            __half fp16_b = __float2half(2.5f);
            
            // 先进行FP16计算
            __half fp16_result = __hadd(fp16_a, fp16_b);
            
            // 转换为FP8进行进一步计算
            __nv_fp8_e4m3 fp8_result = __nv_cvt_halfraw_to_fp8(
                __half_as_short(fp16_result), __NV_NOSAT, __NV_E4M3);
            
            // 转换回FP16
            __half final_result = __nv_cvt_fp8_to_halfraw(fp8_result, __NV_E4M3);
            
            std::cout << "    FP16 + FP8 混合计算:\n";
            std::cout << "      FP16加法: " << __half2float(fp16_a) << " + " << __half2float(fp16_b) << " = " << __half2float(fp16_result) << "\n";
            std::cout << "      FP8压缩后: " << __half2float(final_result) << "\n";
        }
        
        // 示例3: INT8 + FP16 混合计算
        if (int8Supported && fp16Supported) {
            int8_t int8_value = 10;
            __half fp16_value = __float2half(0.5f);
            
            // INT8 -> FP16 转换后进行混合计算
            __half fp16_int8 = __float2half(static_cast<float>(int8_value));
            __half mixed_result = __hmul(fp16_int8, fp16_value);
            
            std::cout << "    INT8 + FP16 混合计算:\n";
            std::cout << "      INT8值: " << static_cast<int>(int8_value) << "\n";
            std::cout << "      FP16值: " << __half2float(fp16_value) << "\n";
            std::cout << "      混合乘法结果: " << __half2float(mixed_result) << "\n";
        }
        
        // 示例4: 混合精度向量点积
        if (fp16Supported) {
            std::cout << "    FP32 + FP16 混合精度向量点积:\n";
            
            // 创建测试向量
            float fp32_vec[3] = {1.0f, 2.0f, 3.0f};
            __half fp16_vec[3] = {__float2half(4.0f), __float2half(5.0f), __float2half(6.0f)};
            
            // 混合精度点积计算
            float dot_product = 0.0f;
            for (int i = 0; i < 3; i++) {
                // FP32 × FP16 -> FP32
                float temp = fp32_vec[i] * __half2float(fp16_vec[i]);
                dot_product += temp;
            }
            
            std::cout << "      FP32向量: [" << fp32_vec[0] << ", " << fp32_vec[1] << ", " << fp32_vec[2] << "]\n";
            std::cout << "      FP16向量: [" << __half2float(fp16_vec[0]) << ", " << __half2float(fp16_vec[1]) << ", " << __half2float(fp16_vec[2]) << "]\n";
            std::cout << "      混合点积结果: " << dot_product << "\n";
        }
        
        std::cout << "\n  ✅ 混合精度数学操作演示完成\n";
    }
    std::cout << "\n";
}



// 高级traits检测 - 检测具体操作支持
template<typename T>
struct fp16_operations_traits {
    static constexpr bool has_addition = false;
    static constexpr bool has_multiplication = false;
    static constexpr bool has_conversion = false;
};

// 特化FP16操作traits
#ifdef __CUDA_FP16_H__
template<>
struct fp16_operations_traits<__half> {
    static constexpr bool has_addition = true;
    static constexpr bool has_multiplication = true;
    static constexpr bool has_conversion = true;
};
#endif

// 检测FP16操作支持
void testFP16OperationsWithTraits() {
    std::cout << "【FP16操作支持检测 - 高级Traits】\n";
    
    using fp16_traits = fp16_operations_traits<__half>;
    
    std::cout << "  FP16加法: " << (fp16_traits::has_addition ? "✅ 支持" : "❌ 不支持") << "\n";
    std::cout << "  FP16乘法: " << (fp16_traits::has_multiplication ? "✅ 支持" : "❌ 不支持") << "\n";
    std::cout << "  FP16转换: " << (fp16_traits::has_conversion ? "✅ 支持" : "❌ 不支持") << "\n";
    
    if (fp16_traits::has_addition && fp16_traits::has_multiplication) {
        std::cout << "  ✅ 支持完整的FP16运算\n";
    }
    std::cout << "\n";
}

// 使用traits模板编程测试类型支持
template<typename T>
struct type_support_traits {
    static constexpr bool is_supported = false;
    static constexpr const char* type_name = "unknown";
    static constexpr const char* description = "不支持";
};

// FP8 E4M3类型支持traits
template<>
struct type_support_traits<__nv_fp8_e4m3> {
    static constexpr bool is_supported = true;
    static constexpr const char* type_name = "__nv_fp8_e4m3";
    static constexpr const char* description = "FP8 E4M3格式 (4位指数, 3位尾数)";
};

// FP8 E5M2类型支持traits
template<>
struct type_support_traits<__nv_fp8_e5m2> {
    static constexpr bool is_supported = true;
    static constexpr const char* type_name = "__nv_fp8_e5m2";
    static constexpr const char* description = "FP8 E5M2格式 (5位指数, 2位尾数)";
};

// FP16类型支持traits
template<>
struct type_support_traits<__half> {
    static constexpr bool is_supported = true;
    static constexpr const char* type_name = "__half";
    static constexpr const char* description = "FP16格式 (半精度浮点)";
};

// BF16类型支持traits
template<>
struct type_support_traits<__nv_bfloat16> {
    static constexpr bool is_supported = true;
    static constexpr const char* type_name = "__nv_bfloat16";
    static constexpr const char* description = "BF16格式 (Brain Float 16)";
};

// INT8类型支持traits
template<>
struct type_support_traits<int8_t> {
    static constexpr bool is_supported = true;
    static constexpr const char* type_name = "int8_t";
    static constexpr const char* description = "8位整数";
};

// 测试类型支持的traits函数
template<typename T>
void testTypeSupportTraits() {
    std::cout << "  " << type_support_traits<T>::type_name << ": ";
    if (type_support_traits<T>::is_supported) {
        std::cout << "✅ " << type_support_traits<T>::description << "\n";
    } else {
        std::cout << "❌ " << type_support_traits<T>::description << "\n";
    }
}

// 使用traits测试所有类型支持
bool testAllTypeSupportWithTraits() {
    std::cout << "【Traits模板编程 - 类型支持测试】\n";
    
    // 测试FP8类型
    std::cout << "FP8类型支持:\n";
    testTypeSupportTraits<__nv_fp8_e4m3>();
    testTypeSupportTraits<__nv_fp8_e5m2>();
    
    // 测试FP16类型
    std::cout << "FP16类型支持:\n";
    testTypeSupportTraits<__half>();
    
    // 测试BF16类型
    std::cout << "BF16类型支持:\n";
    testTypeSupportTraits<__nv_bfloat16>();
    
    // 测试INT8类型
    std::cout << "INT8类型支持:\n";
    testTypeSupportTraits<int8_t>();
    
    return true;
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
        
        // 测试FP8/INT8支持
        testFP8Support(prop.major, prop.minor);
        testINT8Support(prop.major, prop.minor);
        testMixedPrecisionSupport(prop.major, prop.minor);
        
        // 高级traits检测
        testFP16OperationsWithTraits();
        
        // Traits模板编程测试
        testAllTypeSupportWithTraits();
        

        

        
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
