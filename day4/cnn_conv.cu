#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cstring>

// Constant definitions
const int TILE_H = 16;
const int TILE_W = 16;
const int PADDING = 1;
const int KERNEL_SIZE = 9;  // 3x3 kernel

// Basic 2D convolution kernel
__global__ void conv2dBasic(float *input, float *kernel, float *output,
                           int H, int W, int C, int Kh, int Kw, int F,
                           int stride, int padding) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int f = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (h < H && w < W && f < F) {
        float sum = 0.0f;
        
        for (int kh = 0; kh < Kh; kh++) {
            for (int kw = 0; kw < Kw; kw++) {
                for (int c = 0; c < C; c++) {
                    int ih = h * stride + kh - padding;
                    int iw = w * stride + kw - padding;
                    
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        sum += input[c * H * W + ih * W + iw] * 
                               kernel[f * C * Kh * Kw + c * Kh * Kw + kh * Kw + kw];
                    }
                }
            }
        }
        
        output[f * H * W + h * W + w] = sum;
    }
}

// Shared memory optimized 2D convolution kernel
__global__ void conv2dShared(float *input, float *kernel, float *output,
                            int H, int W, int C, int Kh, int Kw, int F,
                            int stride, int padding) {
    __shared__ float s_input[TILE_H + 2*PADDING][TILE_W + 2*PADDING];
    __shared__ float s_kernel[KERNEL_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int h = by * TILE_H + ty;
    int w = bx * TILE_W + tx;
    
    // Cooperatively load input data into shared memory
    for (int c = 0; c < C; c++) {
        // Load current tile data
        if (ty < TILE_H && tx < TILE_W) {
            int ih = h * stride - padding;
            int iw = w * stride - padding;
            
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                s_input[ty + PADDING][tx + PADDING] = 
                    input[c * H * W + ih * W + iw];
            } else {
                s_input[ty + PADDING][tx + PADDING] = 0.0f;
            }
        }
        
        // Load convolution kernel data
        if (ty < Kh && tx < Kw) {
            s_kernel[ty * Kw + tx] = kernel[c * Kh * Kw + ty * Kw + tx];
        }
        
        __syncthreads();
        
        // Calculate convolution
        if (h < H && w < W) {
            float sum = 0.0f;
            for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                    sum += s_input[ty + kh][tx + kw] * s_kernel[kh * Kw + kw];
                }
            }
            output[c * H * W + h * W + w] += sum;
        }
        
        __syncthreads();
    }
}

// Separable convolution: horizontal direction
__global__ void conv2dHorizontal(float *input, float *kernel, float *output,
                                int H, int W, int C, int Kw) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < H && w < W) {
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int kw = 0; kw < Kw; kw++) {
                int iw = w + kw - Kw/2;
                if (iw >= 0 && iw < W) {
                    sum += input[c * H * W + h * W + iw] * kernel[kw];
                }
            }
            output[c * H * W + h * W + w] = sum;
        }
    }
}

// Separable convolution: vertical direction
__global__ void conv2dVertical(float *input, float *kernel, float *output,
                              int H, int W, int C, int Kh) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h < H && w < W) {
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int kh = 0; kh < Kh; kh++) {
                int ih = h + kh - Kh/2;
                if (ih >= 0 && ih < H) {
                    sum += input[c * H * W + ih * W + w] * kernel[kh];
                }
            }
            output[c * H * W + h * W + w] = sum;
        }
    }
}

// Add bias term
__global__ void addBias(float *output, float *bias, int H, int W, int F) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int f = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (h < H && w < W && f < F) {
        int idx = f * H * W + h * W + w;
        output[idx] += bias[f];
    }
}

// Verify results
void verifyResults(float *h_output, float *h_expected, int size) {
    float maxError = 0.0f;
    for (int i = 0; i < size; i++) {
        float error = fabs(h_output[i] - h_expected[i]);
        maxError = max(maxError, error);
    }
    printf("Max error: %.6f\n", maxError);
}

// Performance test function
void performanceTest() {
    const int H = 512, W = 512, C = 64, F = 128;
    const int Kh = 3, Kw = 3;
    const int stride = 1, padding = 1;
    
    // Allocate host memory
    size_t inputSize = C * H * W * sizeof(float);
    size_t kernelSize = F * C * Kh * Kw * sizeof(float);
    size_t outputSize = F * H * W * sizeof(float);
    size_t biasSize = F * sizeof(float);
    
    float *h_input = new float[C * H * W];
    float *h_kernel = new float[F * C * Kh * Kw];
    float *h_output = new float[F * H * W];
    float *h_bias = new float[F];
    
    // Initialize data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < C * H * W; i++) h_input[i] = dis(gen);
    for (int i = 0; i < F * C * Kh * Kw; i++) h_kernel[i] = dis(gen);
    for (int i = 0; i < F; i++) h_bias[i] = dis(gen);
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output, *d_bias;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaMalloc(&d_output, outputSize);
    cudaMalloc(&d_bias, biasSize);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, biasSize, cudaMemcpyHostToDevice);
    
    // Kernel configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, 
                  (H + blockDim.y - 1) / blockDim.y, F);
    
    // Warm up
    conv2dBasic<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                       H, W, C, Kh, Kw, F, stride, padding);
    
    // Performance test
    int iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        conv2dBasic<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                           H, W, C, Kh, Kw, F, stride, padding);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Basic convolution: %.3f ms per iteration\n", milliseconds / iterations);
    printf("Throughput: %.2f GFLOP/s\n", 
           (2.0f * C * Kh * Kw * H * W * F) / (milliseconds / iterations * 1e6));
    
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_bias);
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;
    delete[] h_bias;
}

int main() {
    printf("=== CNN Convolution Performance Test ===\n");
    performanceTest();
    printf("Test completed!\n");
    return 0;
}
