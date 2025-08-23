#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cstring>

// 常量定义
const int TILE_H = 16;
const int TILE_W = 16;
const int PADDING = 1;
const int KERNEL_SIZE = 9;  // 3x3 kernel

// 基础2D卷积kernel
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

// 共享内存优化的2D卷积kernel
__global__ void conv2dShared(float *input, float *kernel, float *output,
                            int H, int W, int C, int Kh, int Kw, int F,
                            int stride, int padding) {
    __shared__ float s_input[TILE_H + 2*PADDING][TILE_W + 2*PADDING];
    __shared__ float s_kernel[KERNEL_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int h = by * TILE_H + ty;
    int w = bx * TILE_W + tx;
    
    // 协作加载输入数据到共享内存
    for (int c = 0; c < C; c++) {
        // 加载当前tile的数据
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
        
        // 加载卷积核数据
        if (ty < Kh && tx < Kw) {
            s_kernel[ty * Kw + tx] = kernel[c * Kh * Kw + ty * Kw + tx];
        }
        
        __syncthreads();
        
        // 计算卷积
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

// 分离卷积：水平方向
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

// 分离卷积：垂直方向
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

// 添加偏置项
__global__ void addBias(float *output, float *bias, int H, int W, int C) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (h < H && w < W && c < C) {
        output[c * H * W + h * W + w] += bias[c];
    }
}

// 验证结果
bool verifyConvolution(float *input, float *kernel, float *output,
                      int H, int W, int C, int Kh, int Kw, int F,
                      int stride, int padding) {
    for (int f = 0; f < F; f++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float expected = 0.0f;
                
                for (int kh = 0; kh < Kh; kh++) {
                    for (int kw = 0; kw < Kw; kw++) {
                        for (int c = 0; c < C; c++) {
                            int ih = h * stride + kh - padding;
                            int iw = w * stride + kw - padding;
                            
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                expected += input[c * H * W + ih * W + iw] * 
                                           kernel[f * C * Kh * Kw + c * Kh * Kw + kh * Kw + kw];
                            }
                        }
                    }
                }
                
                if (abs(output[f * H * W + h * W + w] - expected) > 1e-3) {
                    std::cout << "Mismatch at [" << f << "][" << h << "][" << w << "]: "
                              << output[f * H * W + h * W + w] << " vs " << expected << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

// 性能测试函数
void benchmarkConvolution(int H, int W, int C, int Kh, int Kw, int F,
                         int stride, int padding, int iterations = 10) {
    size_t inputSize = H * W * C * sizeof(float);
    size_t kernelSize = F * C * Kh * Kw * sizeof(float);
    size_t outputSize = H * W * F * sizeof(float);
    size_t biasSize = F * sizeof(float);
    
    // 分配主机内存
    float *h_input = new float[H * W * C];
    float *h_kernel = new float[F * C * Kh * Kw];
    float *h_output = new float[H * W * F];
    float *h_bias = new float[F];
    float *h_output_ref = new float[H * W * F];
    
    // 初始化数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < H * W * C; i++) h_input[i] = dis(gen);
    for (int i = 0; i < F * C * Kh * Kw; i++) h_kernel[i] = dis(gen);
    for (int i = 0; i < F; i++) h_bias[i] = dis(gen);
    
    // 分配设备内存
    float *d_input, *d_kernel, *d_output, *d_bias;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaMalloc(&d_output, outputSize);
    cudaMalloc(&d_bias, biasSize);
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, biasSize, cudaMemcpyHostToDevice);
    
    // 配置kernel参数
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);
    
    // 测试基础版本
    std::cout << "Testing Basic Convolution..." << std::endl;
    cudaMemset(d_output, 0, outputSize);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        conv2dBasic<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                          H, W, C, Kh, Kw, F, stride, padding);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 复制结果回主机
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    // 计算性能指标
    double ops = 2.0 * H * W * C * Kh * Kw * F; // 乘法和加法
    double gflops = (ops * iterations) / (duration.count() * 1000.0);
    
    std::cout << "Basic: " << gflops << " GFLOPS, "
              << duration.count() / iterations << " μs per iteration" << std::endl;
    
    // 测试共享内存版本
    std::cout << "Testing Shared Memory Convolution..." << std::endl;
    cudaMemset(d_output, 0, outputSize);
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        conv2dShared<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                           H, W, C, Kh, Kw, F, stride, padding);
    }
    cudaDeviceSynchronize();
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    gflops = (ops * iterations) / (duration.count() * 1000.0);
    std::cout << "Shared: " << gflops << " GFLOPS, "
              << duration.count() / iterations << " μs per iteration" << std::endl;
    
    // 测试分离卷积版本
    if (Kh == Kw && Kh == 3) {  // 只对3x3卷积核测试分离卷积
        std::cout << "Testing Separable Convolution..." << std::endl;
        cudaMemset(d_output, 0, outputSize);
        
        // 创建1D卷积核
        float *h_kernel_h = new float[3];
        float *h_kernel_v = new float[3];
        float *d_kernel_h, *d_kernel_v;
        
        // 简单的3x3高斯核分解
        h_kernel_h[0] = 0.25f; h_kernel_h[1] = 0.5f; h_kernel_h[2] = 0.25f;
        h_kernel_v[0] = 0.25f; h_kernel_v[1] = 0.5f; h_kernel_v[2] = 0.25f;
        
        cudaMalloc(&d_kernel_h, 3 * sizeof(float));
        cudaMalloc(&d_kernel_v, 3 * sizeof(float));
        cudaMemcpy(d_kernel_h, h_kernel_h, 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel_v, h_kernel_v, 3 * sizeof(float), cudaMemcpyHostToDevice);
        
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            // 水平方向卷积
            conv2dHorizontal<<<gridDim, blockDim>>>(d_input, d_kernel_h, d_output, H, W, C, 3);
            // 垂直方向卷积
            conv2dVertical<<<gridDim, blockDim>>>(d_output, d_kernel_v, d_output, H, W, C, 3);
        }
        cudaDeviceSynchronize();
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        gflops = (ops * iterations) / (duration.count() * 1000.0);
        std::cout << "Separable: " << gflops << " GFLOPS, "
                  << duration.count() / iterations << " μs per iteration" << std::endl;
        
        // 清理资源
        cudaFree(d_kernel_h);
        cudaFree(d_kernel_v);
        delete[] h_kernel_h;
        delete[] h_kernel_v;
    }
    
    // 验证结果
    std::cout << "Verifying results..." << std::endl;
    if (verifyConvolution(h_input, h_kernel, h_output, H, W, C, Kh, Kw, F, stride, padding)) {
        std::cout << "Results verified successfully!" << std::endl;
    } else {
        std::cout << "Results verification failed!" << std::endl;
    }
    
    // 清理资源
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_bias);
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;
    delete[] h_bias;
    delete[] h_output_ref;
}

int main() {
    std::cout << "CUDA CNN Convolution Benchmark" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // 测试不同配置的卷积
    struct ConvConfig {
        int H, W, C, Kh, Kw, F, stride, padding;
        const char* name;
    };
    
    ConvConfig configs[] = {
        {64, 64, 3, 3, 3, 64, 1, 1, "64x64x3 -> 64x64x64 (3x3)"},
        {128, 128, 64, 3, 3, 128, 1, 1, "128x128x64 -> 128x128x128 (3x3)"},
        {224, 224, 3, 5, 5, 64, 1, 2, "224x224x3 -> 224x224x64 (5x5)"},
        {512, 512, 128, 7, 7, 256, 1, 3, "512x512x128 -> 512x512x256 (7x7)"}
    };
    
    for (const auto& config : configs) {
        std::cout << "\nTesting " << config.name << ":" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        benchmarkConvolution(config.H, config.W, config.C, config.Kh, config.Kw, 
                           config.F, config.stride, config.padding);
    }
    
    return 0;
}
