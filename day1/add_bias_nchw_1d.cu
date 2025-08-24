// nvcc add_bias_nchw_1d.cu -o add_bias_nchw_1d && ./add_bias_nchw_1d
#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_bias_nchw_1d(const float* __restrict__ in,
                              float* __restrict__ out,
                              int N, int C, int H, int W,
                              float bias) {
    // 总元素个数
    const long long total = (long long)N * C * H * W;

    // grid-stride loop：让任意网格规模都能覆盖任意大小的张量
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         tid < total;
         tid += (long long)blockDim.x * gridDim.x) {

        // 将线性 tid 映射回 (n,c,h,w)
        int w = tid % W;
        long long t = tid / W;

        int h = t % H;
        t = t / H;

        int c = t % C;
        int n = t / C;

        // 按 NCHW 连续内存计算线性下标
        long long idx = ((long long)n * C + c) * H * W + (long long)h * W + w;

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

    // 启动配置：常见做法 256 线程/块，按需给足 block 数
    int threadsPerBlock = 256;
    int numBlocks = (int)((total + threadsPerBlock - 1) / threadsPerBlock);
    // 给 numBlocks 一个上限，避免过多小 block 带来调度开销
    numBlocks = min(numBlocks, 65535);

    float bias = 1.5f;
    add_bias_nchw_1d<<<numBlocks, threadsPerBlock>>>(d_in, d_out, N, C, H, W, bias);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    // 简单检查几个元素
    for (int i = 0; i < 3; ++i) {
        printf("out[%d] = %.1f\n", i, h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    return 0;
}