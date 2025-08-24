// nvcc add_bias_nchw_3d.cu -o add_bias_nchw_3d && ./add_bias_nchw_3d
#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_bias_nchw_3d(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int N, int C, int H, int W,
                                 float bias) {
    // 使用3D block，每个线程处理一个元素
    // threadIdx.x 对应 W 维度
    // threadIdx.y 对应 H 维度  
    // threadIdx.z 对应 C 维度
    int w = threadIdx.x;
    int h = threadIdx.y;
    int c = threadIdx.z;
    
    // 计算当前block在grid中的位置
    int block_w = blockIdx.x;
    int block_h = blockIdx.y;
    int block_c = blockIdx.z;
    
    // 计算全局的W、H、C坐标
    int global_w = block_w * blockDim.x + w;
    int global_h = block_h * blockDim.y + h;
    int global_c = block_c * blockDim.z + c;
    
    // 检查边界
    if (global_w >= W || global_h >= H || global_c >= C) return;
    
    // 遍历N维度
    for (int n = 0; n < N; ++n) {
        // 计算1D内存索引
        long long idx = ((long long)n * C + global_c) * H * W + (long long)global_h * W + global_w;
        out[idx] = in[idx] + bias;
    }
}

int main() {
    int N = 2, C = 3, H = 4, W = 5; // 随便的尺寸
    long long total = (long long)N * C * H * W;

    // 分配并初始化主机数据
    float *h_in = new float[total];
    float *h_out = new float[total];
    for (long long i = 0; i < total; ++i) h_in[i] = (float)i;

    // 分配设备内存
    float *d_in, *d_out;
    cudaMalloc(&d_in,  total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMemcpy(d_in, h_in, total * sizeof(float), cudaMemcpyHostToDevice);

    // 3D block配置：8x8x4 = 256个线程
    // 注意：blockDim.z ≤ 64，所以用4
    dim3 threadsPerBlock(8, 8, 4);  // blockDim.x=8, blockDim.y=8, blockDim.z=4
    
    // 3D grid配置：按W、H、C维度划分
    int blocks_w = (W + threadsPerBlock.x - 1) / threadsPerBlock.x;  // 向上取整
    int blocks_h = (H + threadsPerBlock.y - 1) / threadsPerBlock.y;  // 向上取整
    int blocks_c = (C + threadsPerBlock.z - 1) / threadsPerBlock.z;  // 向上取整
    dim3 gridSize(blocks_w, blocks_h, blocks_c);
    
    printf("使用3D block配置：\n");
    printf("Block大小: %dx%dx%d = %d线程\n", 
           threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z);
    printf("Grid大小: %dx%dx%d = %d个block\n", 
           gridSize.x, gridSize.y, gridSize.z, gridSize.x * gridSize.y * gridSize.z);
    printf("总线程数: %d\n", threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * gridSize.x * gridSize.y * gridSize.z);
    
    printf("\n维度映射：\n");
    printf("threadIdx.x (8) → W维度 (5)\n");
    printf("threadIdx.y (8) → H维度 (4)\n");
    printf("threadIdx.z (4) → C维度 (3)\n");

    float bias = 1.5f;
    add_bias_nchw_3d<<<gridSize, threadsPerBlock>>>(d_in, d_out, N, C, H, W, bias);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    // 简单检查几个元素
    printf("\n运行结果：\n");
    for (int i = 0; i < 3; ++i) {
        printf("out[%d] = %.1f\n", i, h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    return 0;
}
