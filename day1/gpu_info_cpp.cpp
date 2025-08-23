#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <unistd.h>

// Format bytes to human readable format
std::string formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = bytes;
    
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }
    
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit]);
    return std::string(buffer);
}

// Read GPU information from file
std::string readGPUInfo(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return "Unable to read";
    }
    
    std::string line;
    std::getline(file, line);
    return line;
}

// Get GPU model
std::string getGPUModel() {
    std::string model = readGPUInfo("/sys/class/drm/card0/device/gpu_name");
    if (model == "Unable to read") {
        // Try alternative methods
        model = readGPUInfo("/proc/driver/nvidia/gpus/*/information");
        if (model == "Unable to read") {
            model = readGPUInfo("/sys/class/drm/card0/device/name");
        }
    }
    
    if (model.empty() || model == "Unable to read") {
        return "Unknown";
    }
    
    return model;
}

// Get NVIDIA driver information
std::string getNVIDIADriver() {
    std::string driver = readGPUInfo("/proc/driver/nvidia/version");
    if (driver == "Unable to read") {
        return "Unable to get";
    }
    
    return driver;
}

// Get GPU memory information
void getGPUMemoryInfo() {
    std::cout << "[GPU Memory Information]\n";
    
    // Try to read nvidia-smi output
    std::cout << "  Note: Pure C++ cannot directly access GPU hardware information\n";
    std::cout << "  Recommended commands for detailed information:\n";
    std::cout << "    nvidia-smi\n";
    std::cout << "    nvidia-smi -q -d MEMORY\n";
    std::cout << "    cat /proc/driver/nvidia/gpus/*/information\n";
}

// Get system information
void getSystemInfo() {
    std::cout << "[System Information]\n";
    
    // CPU info
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                size_t pos = line.find(": ");
                if (pos != std::string::npos) {
                    std::cout << "  CPU: " << line.substr(pos + 2) << "\n";
                    break;
                }
            }
        }
    }
    
    // Memory info
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
                        try {
                            size_t memKB = std::stoull(memStr);
                            std::cout << "  Memory: " << formatBytes(memKB * 1024) << "\n";
                        } catch (...) {
                            std::cout << "  Memory: " << memStr << " kB\n";
                        }
                    }
                    break;
                }
            }
        }
    }
    
    // OS info
    std::ifstream osrelease("/etc/os-release");
    if (osrelease.is_open()) {
        std::string line;
        while (std::getline(osrelease, line)) {
            if (line.find("PRETTY_NAME") != std::string::npos) {
                size_t pos = line.find("=");
                if (pos != std::string::npos) {
                    std::string osName = line.substr(pos + 1);
                    if (osName.front() == '"' && osName.back() == '"') {
                        osName = osName.substr(1, osName.length() - 2);
                    }
                    std::cout << "  OS: " << osName << "\n";
                    break;
                }
            }
        }
    }
    
    // Kernel version
    std::ifstream version("/proc/version");
    if (version.is_open()) {
        std::string line;
        std::getline(version, line);
        size_t pos = line.find("Linux version ");
        if (pos != std::string::npos) {
            size_t endPos = line.find(" ", pos + 14);
            if (endPos != std::string::npos) {
                std::cout << "  Kernel: " << line.substr(pos + 14, endPos - pos - 14) << "\n";
            }
        }
    }
}

// Check CUDA environment
void checkCUDAEnvironment() {
    std::cout << "[CUDA Environment]\n";
    
    // Check CUDA_HOME
    const char* cudaHome = getenv("CUDA_HOME");
    if (cudaHome) {
        std::cout << "  CUDA_HOME: " << cudaHome << "\n";
    } else {
        std::cout << "  CUDA_HOME: Not set\n";
    }
    
    // Check PATH for nvcc
    const char* path = getenv("PATH");
    if (path) {
        std::string pathStr(path);
        if (pathStr.find("/usr/local/cuda/bin") != std::string::npos) {
            std::cout << "  nvcc: Available in /usr/local/cuda/bin\n";
        } else if (pathStr.find("/usr/bin") != std::string::npos) {
            std::cout << "  nvcc: Available in /usr/bin\n";
        } else {
            std::cout << "  nvcc: Not found in PATH\n";
        }
    }
    
    // Check for CUDA libraries
    std::cout << "  CUDA Libraries:\n";
    
    const char* libPaths[] = {
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib"
    };
    
    for (const char* libPath : libPaths) {
        std::ifstream libFile(std::string(libPath) + "/libcuda.so");
        if (libFile.good()) {
            std::cout << "    libcuda.so: " << libPath << "\n";
            break;
        }
    }
    
    for (const char* libPath : libPaths) {
        std::ifstream libFile(std::string(libPath) + "/libcudart.so");
        if (libFile.good()) {
            std::cout << "    libcudart.so: " << libPath << "\n";
            break;
        }
    }
}

int main() {
    std::cout << "=== GPU Information (Pure C++ Version) ===\n";
    std::cout << "Note: This program can only access system information\n";
    std::cout << "      For detailed GPU info, use nvidia-smi or CUDA programs\n\n";
    
    getSystemInfo();
    std::cout << "\n";
    
    getGPUMemoryInfo();
    std::cout << "\n";
    
    checkCUDAEnvironment();
    std::cout << "\n";
    
    std::cout << "=== Recommendations ===\n";
    std::cout << "1. Use 'nvidia-smi' for real-time GPU status\n";
    std::cout << "2. Use 'nvidia-smi -q' for detailed GPU information\n";
    std::cout << "3. Use CUDA programs (gpu_info.cu) for hardware capabilities\n";
    std::cout << "4. Use Python with nvidia-ml-py for programmatic access\n";
    
    return 0;
}
