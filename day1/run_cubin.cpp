#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <memory>

// RAII wrapper for CUDA resources
class CUDADevice {
private:
    CUdevice device_;
    CUcontext context_;
    
public:
    CUDADevice() {
        CUresult result = cuInit(0);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("cuInit failed");
        }
        
        result = cuDeviceGet(&device_, 0);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("cuDeviceGet failed");
        }
        
        result = cuCtxCreate(&context_, 0, device_);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("cuCtxCreate failed");
        }
    }
    
    ~CUDADevice() {
        if (context_) {
            cuCtxDestroy(context_);
        }
    }
    
    CUcontext getContext() const { return context_; }
    CUdevice getDevice() const { return device_; }
    
    // Disable copying
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
            throw std::runtime_error("Failed to load CUBIN file: " + cubin_file);
        }
    }
    
    ~CUDAModule() {
        if (module_) {
            cuModuleUnload(module_);
        }
    }
    
    CUmodule get() const { return module_; }
    
    // Disable copying
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
            throw std::runtime_error("Device memory allocation failed");
        }
    }
    
    ~CUDAMemory() {
        if (ptr_) {
            cuMemFree(ptr_);
        }
    }
    
    CUdeviceptr get() const { return ptr_; }
    size_t size() const { return size_; }
    
    // Disable copying
    CUDAMemory(const CUDAMemory&) = delete;
    CUDAMemory& operator=(const CUDAMemory&) = delete;
};

int main(int argc, char *argv[]) {
    try {
        // Check command line arguments
        if (argc != 2) {
            std::cout << "Usage: " << argv[0] << " <cubin_file>" << std::endl;
            std::cout << "Example: " << argv[0] << " vector_add.cubin" << std::endl;
            return 1;
        }
        
        std::string cubin_file = argv[1];
        
        // Determine kernel name based on CUBIN filename
        std::string kernel_name;
        if (cubin_file.find("vector_add") != std::string::npos) {
            kernel_name = "vector_add";
        } else if (cubin_file.find("vector_dot") != std::string::npos) {
            kernel_name = "vector_dot";
        } else {
            std::cout << "Error: Unsupported CUBIN file: " << cubin_file << std::endl;
            std::cout << "Supported formats: vector_add.cubin or vector_dot.cubin" << std::endl;
            return 1;
        }
        
        std::cout << "Loading CUBIN file: " << cubin_file << std::endl;
        std::cout << "Kernel name: " << kernel_name << std::endl;
        
        // Initialize CUDA device
        CUDADevice device;
        
        // Load CUBIN module
        CUDAModule module(cubin_file);
        
        // Get kernel function
        CUfunction kernel;
        CUresult result = cuModuleGetFunction(&kernel, module.get(), kernel_name.c_str());
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get kernel function: " + kernel_name);
        }
        
        std::cout << "Kernel loaded successfully!" << std::endl;
        
        // Prepare test data
        const int n = 3;
        std::vector<float> h_a = {1.0f, 2.0f, 3.0f};
        std::vector<float> h_b = {2.0f, 2.0f, 2.0f};
        std::vector<float> h_c(n, 0.0f);
        
        // Allocate device memory
        CUDAMemory d_a(n * sizeof(float));
        CUDAMemory d_b(n * sizeof(float));
        CUDAMemory d_c(n * sizeof(float));
        
        // Copy data to device
        cuMemcpyHtoD(d_a.get(), h_a.data(), n * sizeof(float));
        cuMemcpyHtoD(d_b.get(), h_b.data(), n * sizeof(float));
        cuMemcpyHtoD(d_c.get(), h_c.data(), n * sizeof(float));
        
        // Launch kernel
        std::cout << "Launching kernel..." << std::endl;
        
        // Prepare kernel arguments
        int n_param = n;  // Create non-const copy
        CUdeviceptr d_a_ptr = d_a.get();
        CUdeviceptr d_b_ptr = d_b.get();
        CUdeviceptr d_c_ptr = d_c.get();
        
        void* args[] = {&d_a_ptr, &d_b_ptr, &d_c_ptr, &n_param};
        
        // Launch kernel with 1 block of 256 threads
        result = cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Kernel launch failed");
        }
        
        std::cout << "Kernel execution completed!" << std::endl;
        
        // Copy results back to host
        cuMemcpyDtoH(h_c.data(), d_c.get(), n * sizeof(float));
        
        // Display results
        std::cout << "Results:" << std::endl;
        if (kernel_name == "vector_add") {
            std::cout << "c[0] = " << h_c[0] << " (expected: " << h_a[0] + h_b[0] << ")" << std::endl;
            std::cout << "c[1] = " << h_c[1] << " (expected: " << h_a[1] + h_b[1] << ")" << std::endl;
            std::cout << "c[2] = " << h_c[2] << " (expected: " << h_a[2] + h_b[2] << ")" << std::endl;
        } else if (kernel_name == "vector_dot") {
            float expected = h_a[0] * h_b[0] + h_a[1] * h_b[1] + h_a[2] * h_b[2];
            std::cout << "c[0] = " << h_c[0] << " (expected: " << expected << ")" << std::endl;
            std::cout << "c[1] = " << h_c[1] << " (expected: 0)" << std::endl;
            std::cout << "c[2] = " << h_c[2] << " (expected: 0)" << std::endl;
        }
        
        std::cout << "Test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


