#include <cuda_runtime.h>
#include <iostream>

// 向量点积CUDA kernel
extern "C" __global__ void vector_dot(const float *a, const float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 每个线程计算一个元素对
        atomicAdd(result, a[idx] * b[idx]);
    }
}

// 主机代码：向量点积测试
int main() {
    const int n = 3;
    
    // 分配主机内存
    float *h_a = new float[n];
    float *h_b = new float[n];
    float h_result = 0.0f;
    
    // 初始化向量a和b
    for (int i = 0; i < n; i++) {
        h_a[i] = i + 1;  // a = [1, 2, 3]
        h_b[i] = 2;      // b = [2, 2, 2]
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));  // 初始化为0
    
    // 设置线程块和网格大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // 启动kernel
    vector_dot<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);
    
    // 复制结果回主机
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // 显示输入和结果
    std::cout << "Vector a: [";
    for (int i = 0; i < n; i++) {
        std::cout << h_a[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Vector b: [";
    for (int i = 0; i < n; i++) {
        std::cout << h_b[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Dot product a · b = " << h_result << std::endl;
    
    // 验证结果：1*2 + 2*2 + 3*2 = 2 + 4 + 6 = 12
    float expected = 0;
    for (int i = 0; i < n; i++) {
        expected += h_a[i] * h_b[i];
    }
    std::cout << "Expected result: " << expected << std::endl;
    
    // 清理内存
    delete[] h_a, h_b;
    cudaFree(d_a), cudaFree(d_b), cudaFree(d_result);
    
    return 0;
}
