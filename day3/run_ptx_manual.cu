#include <cuda.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

// Helper: load PTX file into string
std::string loadPTX(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open PTX file!" << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Host arrays
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N, 0.0f);

    // Initialize CUDA Driver API
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr d_A, d_B, d_C;

    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // Load PTX
    std::string ptxSource = loadPTX("vector_add.ptx");
    CUresult res = cuModuleLoadDataEx(&module, ptxSource.c_str(), 0, nullptr, nullptr);
    if (res != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        std::cerr << "Failed to load PTX! CUDA Error: " << errStr << std::endl;
        return 1;
    }

    // Get kernel handle
    res = cuModuleGetFunction(&kernel, module, "vector_add");
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to get kernel handle!" << std::endl;
        return 1;
    }

    // Allocate GPU memory
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_B, size);
    cuMemAlloc(&d_C, size);

    // Copy inputs to GPU
    cuMemcpyHtoD(d_A, h_A.data(), size);
    cuMemcpyHtoD(d_B, h_B.data(), size);

    // Set up kernel arguments
    void *args[] = { &d_A, &d_B, &d_C, (void*)&N };

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    res = cuLaunchKernel(
        kernel,
        blocksPerGrid, 1, 1,   // Grid
        threadsPerBlock, 1, 1, // Block
        0,                     // Shared mem
        0,                     // Stream
        args,
        nullptr
    );

    if (res != CUDA_SUCCESS) {
        std::cerr << "Kernel launch failed!" << std::endl;
        return 1;
    }

    // Copy result back to host
    cuMemcpyDtoH(h_C.data(), d_C, size);

    // Print first 10 results
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // Cleanup
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}


