// nvcc add_bias_nchw_2d.cu -o add_bias_nchw_2d && ./add_bias_nchw_2d
#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_bias_nchw_2d(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int N, int C, int H, int W,
                                 float bias) {
    // 使用2D block，每个线程处理一个元素
    // threadIdx.x 对应 W 维度，threadIdx.y 对应 H 维度
    int w = threadIdx.x;
    int h = threadIdx.y;
    
    // 计算当前block在grid中的位置
    int block_w = blockIdx.x;
    int block_h = blockIdx.y;
    
    // 计算全局的W和H坐标
    int global_w = block_w * blockDim.x + w;
    int global_h = block_h * blockDim.y + h;
    
    // 检查边界
    if (global_w >= W || global_h >= H) return;
    
    // 遍历N和C维度
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            // 计算1D内存索引
            long long idx = ((long long)n * C + c) * H * W + (long long)global_h * W + global_w;
            out[idx] = in[idx] + bias;
        }
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

    // 2D block配置：16x16 = 256个线程
    dim3 threadsPerBlock(16, 16);  // blockDim.x=16, blockDim.y=16
    
    // 2D grid配置：按W和H维度划分
    int blocks_w = (W + threadsPerBlock.x - 1) / threadsPerBlock.x;  // 向上取整
    int blocks_h = (H + threadsPerBlock.y - 1) / threadsPerBlock.y;  // 向上取整
    dim3 gridSize(blocks_w, blocks_h);
    
    printf("使用2D block配置：\n");
    printf("Block大小: %dx%d = %d线程\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.x * threadsPerBlock.y);
    printf("Grid大小: %dx%d = %d个block\n", gridSize.x, gridSize.y, gridSize.x * gridSize.y);
    printf("总线程数: %d\n", threadsPerBlock.x * threadsPerBlock.y * gridSize.x * gridSize.y);

    float bias = 1.5f;
    add_bias_nchw_2d<<<gridSize, threadsPerBlock>>>(d_in, d_out, N, C, H, W, bias);
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
