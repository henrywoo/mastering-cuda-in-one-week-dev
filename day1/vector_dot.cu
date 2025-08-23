#include <cuda_runtime.h>
#include <iostream>

// Vector dot product CUDA kernel
__global__ void vector_dot(float *a, float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes one element pair
    if (idx < n) {
        atomicAdd(result, a[idx] * b[idx]);
    }
}

int main() {
    const int n = 3;
    
    // Host code: vector dot product test
    float h_a[n], h_b[n], h_result;
    
    // Allocate host memory
    float *d_a, *d_b, *d_result;
    
    // Initialize vectors a and b
    h_a[0] = 1.0f;
    h_a[1] = 2.0f;
    h_a[2] = 3.0f;
    
    h_b[0] = 2.0f;
    h_b[1] = 2.0f;
    h_b[2] = 2.0f;
    
    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));  // Initialize to 0
    
    // Set thread block and grid size
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    vector_dot<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);
    
    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Display input and result
    std::cout << "Vector a: [" << h_a[0] << ", " << h_a[1] << ", " << h_a[2] << "]" << std::endl;
    std::cout << "Vector b: [" << h_b[0] << ", " << h_b[1] << ", " << h_b[2] << "]" << std::endl;
    std::cout << "Dot product result: " << h_result << std::endl;
    
    // Calculate expected result
    float expected = 0.0f;
    for (int i = 0; i < n; i++) {
        expected += h_a[i] * h_b[i];
    }
    std::cout << "Expected result: " << expected << std::endl;
    
    // Verify result: 1*2 + 2*2 + 3*2 = 2 + 4 + 6 = 12
    if (abs(h_result - expected) < 1e-5) {
        std::cout << "Result verification: PASSED" << std::endl;
    } else {
        std::cout << "Result verification: FAILED" << std::endl;
        std::cout << "Difference: " << abs(h_result - expected) << std::endl;
    }
    
    // Clean up memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    return 0;
}
