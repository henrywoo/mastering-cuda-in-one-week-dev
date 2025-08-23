#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
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

// 从文件读取GPU信息
std::string readGPUInfo(const std::string& filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        return content;
    }
    return "无法读取";
}

// 获取GPU型号
std::string getGPUModel() {
    std::string model = readGPUInfo("/proc/driver/nvidia/gpus/*/information");
    if (model == "无法读取") {
        // 尝试其他方法
        std::ifstream file("/proc/cpuinfo");
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (line.find("model name") != std::string::npos) {
                    size_t pos = line.find(": ");
                    if (pos != std::string::npos) {
                        return line.substr(pos + 2);
                    }
                }
            }
            file.close();
        }
        return "未知";
    }
    return model;
}

// 获取NVIDIA驱动信息
std::string getNvidiaDriverVersion() {
    std::ifstream file("/proc/driver/nvidia/version");
    if (file.is_open()) {
        std::string line;
        if (std::getline(file, line)) {
            return line;
        }
        file.close();
    }
    return "无法获取";
}

// 获取GPU内存信息
void getGPUMemoryInfo() {
    std::cout << "【GPU内存信息】\n";
    
    // 尝试读取nvidia-smi输出
    std::cout << "  注意: 纯C++无法直接访问GPU硬件信息\n";
    std::cout << "  建议使用以下命令获取详细信息:\n";
    std::cout << "    nvidia-smi\n";
    std::cout << "    nvidia-smi -q\n";
    std::cout << "    cat /proc/driver/nvidia/gpus/*/information\n\n";
}

// 获取系统信息
void getSystemInfo() {
    std::cout << "【系统信息】\n";
    
    // CPU信息
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                size_t pos = line.find(": ");
                if (pos != std::string::npos) {
                    std::cout << "  CPU型号: " << line.substr(pos + 2) << "\n";
                    break;
                }
            }
        }
        cpuinfo.close();
    }
    
    // 内存信息
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal") != std::string::npos) {
                size_t pos = line.find(": ");
                if (pos != std::string::npos) {
                    std::string memStr = line.substr(pos + 2);
                    size_t kbPos = memStr.find(" kB");
                    if (kbPos != std::string::npos) {
                        memStr = memStr.substr(0, kbPos);
                        long memKB = std::stol(memStr);
                        std::cout << "  系统内存: " << formatBytes(memKB * 1024) << "\n";
                    }
                    break;
                }
            }
        }
        meminfo.close();
    }
    
    // 操作系统信息
    std::ifstream osrelease("/etc/os-release");
    if (osrelease.is_open()) {
        std::string line;
        while (std::getline(osrelease, line)) {
            if (line.find("PRETTY_NAME") != std::string::npos) {
                size_t pos = line.find("=");
                if (pos != std::string::npos) {
                    std::string osName = line.substr(pos + 1);
                    // 移除引号
                    if (osName.front() == '"') osName = osName.substr(1);
                    if (osName.back() == '"') osName = osName.substr(0, osName.length() - 1);
                    std::cout << "  操作系统: " << osName << "\n";
                    break;
                }
            }
        }
        osrelease.close();
    }
    
    std::cout << "\n";
}

// 获取CUDA环境信息
void getCUDAEnvironmentInfo() {
    std::cout << "【CUDA环境信息】\n";
    
    const char* cudaHome = std::getenv("CUDA_HOME");
    const char* cudaPath = std::getenv("CUDA_PATH");
    const char* path = std::getenv("PATH");
    
    std::cout << "  CUDA_HOME: " << (cudaHome ? cudaHome : "未设置") << "\n";
    std::cout << "  CUDA_PATH: " << (cudaPath ? cudaPath : "未设置") << "\n";
    
    if (path) {
        std::string pathStr(path);
        if (pathStr.find("cuda") != std::string::npos) {
            std::cout << "  PATH包含CUDA: 是\n";
        } else {
            std::cout << "  PATH包含CUDA: 否\n";
        }
    }
    
    // 检查CUDA工具
    std::cout << "  CUDA工具检查:\n";
    
    std::vector<std::string> cudaTools = {
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda/bin/nvidia-smi",
        "/usr/bin/nvcc",
        "/usr/bin/nvidia-smi"
    };
    
    for (const auto& tool : cudaTools) {
        std::ifstream file(tool);
        if (file.good()) {
            std::cout << "    " << tool << ": 存在\n";
        } else {
            std::cout << "    " << tool << ": 不存在\n";
        }
        file.close();
    }
    
    std::cout << "\n";
}

// 性能优化建议
void getPerformanceRecommendations() {
    std::cout << "【性能优化建议】\n";
    std::cout << "  1. 使用nvidia-smi监控GPU使用情况\n";
    std::cout << "  2. 使用nvprof或nsight compute进行性能分析\n";
    std::cout << "  3. 测试不同线程块大小找到最优配置\n";
    std::cout << "  4. 监控共享内存和寄存器使用情况\n";
    std::cout << "  5. 使用cuda-memcheck检查内存错误\n";
    std::cout << "  6. 考虑使用CUDA Occupancy Calculator\n";
    std::cout << "  7. 使用nvcc --version查看CUDA版本\n";
    std::cout << "  8. 使用nvidia-smi -q查看详细GPU信息\n\n";
}

int main() {
    std::cout << "=== GPU信息获取程序 (纯C++版本) ===\n\n";
    std::cout << "注意: 纯C++无法直接访问GPU硬件信息\n";
    std::cout << "此程序提供系统信息和环境检查\n\n";
    
    getSystemInfo();
    getCUDAEnvironmentInfo();
    getGPUMemoryInfo();
    getPerformanceRecommendations();
    
    std::cout << "=== 建议使用以下命令获取完整GPU信息 ===\n";
    std::cout << "nvidia-smi                    # 基本GPU状态\n";
    std::cout << "nvidia-smi -q                 # 详细GPU信息\n";
    std::cout << "nvcc --version                # CUDA编译器版本\n";
    std::cout << "cat /proc/driver/nvidia/gpus/*/information  # 系统GPU信息\n";
    std::cout << "lspci | grep -i nvidia       # PCI设备信息\n\n";
    
    std::cout << "=== 程序执行完成 ===\n";
    return 0;
}
