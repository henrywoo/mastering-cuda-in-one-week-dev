#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <memory>

// RAII包装器用于CUDA资源
class CUDADevice {
private:
    CUdevice device_;
    CUcontext context_;
    
public:
    CUDADevice() {
        CUresult result = cuInit(0);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("cuInit失败");
        }
        
        result = cuDeviceGet(&device_, 0);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("cuDeviceGet失败");
        }
        
        result = cuCtxCreate(&context_, 0, device_);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("cuCtxCreate失败");
        }
    }
    
    ~CUDADevice() {
        if (context_) {
            cuCtxDestroy(context_);
        }
    }
    
    CUcontext getContext() const { return context_; }
    CUdevice getDevice() const { return device_; }
    
    // 禁用拷贝
    CUDADevice(const CUDADevice&) = delete;
    CUDADevice& operator=(const CUDADevice&) = delete;
};

class CUDAModule {
private:
    CUmodule module_;
    
public:
    CUDAModule(const std::string& cubin_file) {
        CUresult result = cuModuleLoad(&module_, cubin_file.c_str());
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("无法加载CUBIN文件: " + cubin_file);
        }
    }
    
    ~CUDAModule() {
        if (module_) {
            cuModuleUnload(module_);
        }
    }
    
    CUmodule get() const { return module_; }
    
    // 禁用拷贝
    CUDAModule(const CUDAModule&) = delete;
    CUDAModule& operator=(const CUDAModule&) = delete;
};

class CUDAMemory {
private:
    CUdeviceptr ptr_;
    size_t size_;
    
public:
    CUDAMemory(size_t size) : size_(size) {
        CUresult result = cuMemAlloc(&ptr_, size);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("设备内存分配失败");
        }
    }
    
    ~CUDAMemory() {
        if (ptr_) {
            cuMemFree(ptr_);
        }
    }
    
    CUdeviceptr get() const { return ptr_; }
    size_t size() const { return size_; }
    
    // 禁用拷贝
    CUDAMemory(const CUDAMemory&) = delete;
    CUDAMemory& operator=(const CUDAMemory&) = delete;
};

int main(int argc, char *argv[]) {
    try {
        // 检查命令行参数
        if (argc != 2) {
            std::cout << "用法: " << argv[0] << " <cubin_file>" << std::endl;
            std::cout << "例如: " << argv[0] << " vector_add.cubin" << std::endl;
            std::cout << "      " << argv[0] << " vector_dot.cubin" << std::endl;
            return 1;
        }
        
        std::string cubin_file = argv[1];
        std::string kernel_name;
        
        // 根据CUBIN文件名确定kernel名称
        if (cubin_file.find("vector_add") != std::string::npos) {
            kernel_name = "vector_add";
        } else if (cubin_file.find("vector_dot") != std::string::npos) {
            kernel_name = "vector_dot";
        } else {
            std::cout << "错误: 不支持的CUBIN文件: " << cubin_file << std::endl;
            std::cout << "支持的格式: vector_add.cubin 或 vector_dot.cubin" << std::endl;
            return 1;
        }
        
        std::cout << "加载CUBIN文件: " << cubin_file << std::endl;
        std::cout << "Kernel名称: " << kernel_name << std::endl;
        
        // 初始化CUDA设备
        CUDADevice cuda_device;
        
        // 加载CUBIN模块
        CUDAModule cuda_module(cubin_file);
        
        // 获取kernel函数
        CUfunction kernel;
        CUresult result = cuModuleGetFunction(&kernel, cuda_module.get(), kernel_name.c_str());
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("无法获取kernel函数: " + kernel_name);
        }
        
        std::cout << "Kernel加载成功!" << std::endl;
        
        // 准备测试数据
        const int n = 3;
        std::unique_ptr<float[]> h_a = std::make_unique<float[]>(n);
        std::unique_ptr<float[]> h_b = std::make_unique<float[]>(n);
        std::unique_ptr<float[]> h_c = std::make_unique<float[]>(n);
        
        // 初始化数据
        for (int i = 0; i < n; i++) {
            h_a[i] = i + 1;  // a = [1, 2, 3]
            h_b[i] = 2;      // b = [2, 2, 2]
        }
        
        // 分配设备内存
        CUDAMemory d_a(n * sizeof(float));
        CUDAMemory d_b(n * sizeof(float));
        
        CUDAMemory d_c(kernel_name == "vector_dot" ? sizeof(float) : n * sizeof(float));
        
        // 如果是向量点积，初始化结果为0
        if (kernel_name == "vector_dot") {
            float zero = 0.0f;
            result = cuMemcpyHtoD(d_c.get(), &zero, sizeof(float));
            if (result != CUDA_SUCCESS) {
                throw std::runtime_error("初始化结果失败");
            }
        }
        
        // 复制数据到设备
        result = cuMemcpyHtoD(d_a.get(), h_a.get(), n * sizeof(float));
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("数据复制到设备失败");
        }
        
        result = cuMemcpyHtoD(d_b.get(), h_b.get(), n * sizeof(float));
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("数据复制到设备失败");
        }
        
        // 设置kernel参数
        int n_param = n;
        CUdeviceptr d_a_ptr = d_a.get();
        CUdeviceptr d_b_ptr = d_b.get();
        CUdeviceptr d_c_ptr = d_c.get();
        void* args[] = {&d_a_ptr, &d_b_ptr, &d_c_ptr, &n_param};
        
        // 启动kernel
        std::cout << "启动kernel..." << std::endl;
        result = cuLaunchKernel(kernel, 1, 1, 1,  // grid dimensions
                                            n, 1, 1,  // block dimensions
                                            0, 0,     // shared memory and stream
                                            args, 0); // arguments and extra
        
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("kernel启动失败");
        }
        
        // 复制结果回主机
        if (kernel_name == "vector_dot") {
            // 向量点积：复制单个结果值
            float h_result;
            result = cuMemcpyDtoH(&h_result, d_c.get(), sizeof(float));
            if (result != CUDA_SUCCESS) {
                throw std::runtime_error("结果复制回主机失败");
            }
            
            std::cout << "Kernel执行完成!" << std::endl;
            std::cout << "向量点积结果: " << h_result << std::endl;
            
            // 验证结果
            float expected = 0;
            for (int i = 0; i < n; i++) {
                expected += h_a[i] * h_b[i];
            }
            std::cout << "期望结果: " << expected << " (1×2 + 2×2 + 3×2 = 12)" << std::endl;
            
        } else {
            // 向量加法：复制n个结果值
            result = cuMemcpyDtoH(h_c.get(), d_c.get(), n * sizeof(float));
            if (result != CUDA_SUCCESS) {
                throw std::runtime_error("结果复制回主机失败");
            }
            
            std::cout << "Kernel执行完成!" << std::endl;
            std::cout << "向量加法结果:" << std::endl;
            for (int i = 0; i < n; i++) {
                std::cout << "c[" << i << "] = " << h_c[i] << std::endl;
            }
            
            // 验证结果
            std::cout << "期望结果: [3, 4, 5] ([1,2,3] + [2,2,2])" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


