#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

// Format bytes to human readable format
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

// Format frequency to GHz
std::string formatFrequency(int freqKHz) {
    double freqGHz = freqKHz / 1000000.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << freqGHz << " GHz";
    return oss.str();
}

// Get compute capability string
std::string getComputeCapabilityString(int major, int minor) {
    std::ostringstream oss;
    oss << major << "." << minor;
    return oss.str();
}

// Get GPU architecture name
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

// Get GPU model range based on SM count
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
    std::cout << "=== NVIDIA GPU Detailed Configuration Information ===\n\n";
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "Error: No CUDA-capable GPU devices found\n";
        return -1;
    }
    
    std::cout << "Detected " << deviceCount << " CUDA device(s)\n\n";
    
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        std::cout << "=== Device " << device << ": " << prop.name << " ===\n";
        std::cout << std::string(50, '-') << "\n\n";
        
        // Basic information
        std::cout << "[Basic Information]\n";
        std::cout << "  Device Name: " << prop.name << "\n";
        std::cout << "  Compute Capability: " << getComputeCapabilityString(prop.major, prop.minor) << "\n";
        std::cout << "  Architecture: " << getArchitectureName(prop.major, prop.minor) << "\n";
        std::cout << "  Multi-Processor Count: " << prop.multiProcessorCount << " (SM)\n";
        std::cout << "  Estimated Model: " << getGPUModelRange(prop.multiProcessorCount) << "\n\n";
        
        // Clock frequencies
        std::cout << "[Clock Frequencies]\n";
        std::cout << "  Memory Clock: " << formatFrequency(prop.memoryClockRate) << "\n";
        std::cout << "  GPU Clock: " << formatFrequency(prop.clockRate) << "\n\n";
        
        // Memory information
        std::cout << "[Memory Information]\n";
        std::cout << "  Total Global Memory: " << formatBytes(prop.totalGlobalMem) << "\n";
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
        std::cout << "  Memory Bandwidth: " << formatBytes(prop.memoryBusWidth * prop.memoryClockRate * 2 / 8) << "/s\n\n";
        
        // Thread and block configuration
        std::cout << "[Thread and Block Configuration]\n";
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor << "\n";
        std::cout << "  Warp Size: " << prop.warpSize << "\n\n";
        
        // Block dimensions
        std::cout << "[Block Dimensions]\n";
        std::cout << "  Max Block Dimensions: (" << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
        std::cout << "  Max Grid Dimensions: (" << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n\n";
        
        // Shared memory and registers
        std::cout << "[Shared Memory and Registers]\n";
        std::cout << "  Shared Memory per Block: " << formatBytes(prop.sharedMemPerBlock) << "\n";
        std::cout << "  Shared Memory per SM: " << formatBytes(prop.sharedMemPerMultiprocessor) << "\n";
        std::cout << "  Registers per Block: " << prop.regsPerBlock << "\n";
        std::cout << "  Registers per SM: " << prop.regsPerMultiprocessor << "\n\n";
        
        // Cache information
        std::cout << "[Cache Information]\n";
        std::cout << "  L2 Cache Size: " << formatBytes(prop.l2CacheSize) << "\n";
        std::cout << "  Constant Memory: " << formatBytes(prop.totalConstMem) << "\n\n";
        
        // Compute features
        std::cout << "[Compute Features]\n";
        std::cout << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << "\n";
        std::cout << "  Concurrent Memory Operations: " << (prop.concurrentManagedAccess ? "Yes" : "No") << "\n";
        std::cout << "  Unified Virtual Addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << "\n";
        std::cout << "  Cooperative Launch: " << (prop.cooperativeLaunch ? "Yes" : "No") << "\n";
        std::cout << "  Cooperative Multi-Device Launch: " << (prop.cooperativeMultiDeviceLaunch ? "Yes" : "No") << "\n\n";
        
        // Performance recommendations
        std::cout << "[Performance Recommendations]\n";
        std::cout << "  Optimal Thread Block Size: " << prop.maxThreadsPerBlock << " threads\n";
        std::cout << "  Optimal Threads per SM: " << prop.maxThreadsPerMultiProcessor << " threads\n";
        std::cout << "  Optimal Blocks per SM: " << prop.maxBlocksPerMultiProcessor << " blocks\n";
        std::cout << "  Memory Coalescing: " << (prop.memoryClockRate > 0 ? "Enabled" : "Disabled") << "\n";
        std::cout << "  Shared Memory Bank Conflicts: Avoid multiples of " << prop.warpSize << "\n\n";
        
        // Tensor Core information (if available)
        if (prop.major >= 7) {
            std::cout << "[Tensor Core Information]\n";
            std::cout << "  Tensor Cores: Available\n";
            if (prop.major == 8 && prop.minor >= 0) {
                std::cout << "  FP16 Performance: 2x FP32\n";
                std::cout << "  INT8 Performance: 4x FP32\n";
            } else if (prop.major == 7 && prop.minor >= 5) {
                std::cout << "  FP16 Performance: 2x FP32\n";
            }
            std::cout << "\n";
        }
        
        // CUDA version compatibility
        std::cout << "[CUDA Version Compatibility]\n";
        std::cout << "  Minimum CUDA Version: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Driver Version: " << prop.driverVersion << "\n";
        std::cout << "  Runtime Version: " << prop.runtimeVersion << "\n\n";
        
        std::cout << std::string(50, '=') << "\n\n";
    }
    
    std::cout << "=== Summary ===\n";
    std::cout << "For optimal performance:\n";
    std::cout << "1. Use thread block sizes that are multiples of " << prop.warpSize << "\n";
    std::cout << "2. Ensure total threads per SM doesn't exceed " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "3. Use shared memory efficiently (up to " << formatBytes(prop.sharedMemPerBlock) << " per block)\n";
    std::cout << "4. Consider memory coalescing for global memory access\n";
    std::cout << "5. Use appropriate data types (FP16 for Tensor Cores if available)\n\n";
    
    return 0;
}
