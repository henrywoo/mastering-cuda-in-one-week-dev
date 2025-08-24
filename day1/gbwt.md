# CUDA编程模型：从抽象到现实

让我来聊聊CUDA的编程模型。当你刚开始接触CUDA时，可能会觉得这个概念有点抽象，但理解它对于写出高效的GPU代码至关重要。

## 编程模型的层次结构

CUDA的编程模型可以这样理解：

```
Grid  →  Block  →  Thread
```

简单来说，Grid是一组Block的集合，Block是一组Thread的集合，而Thread就是最小的执行单元——你在kernel函数里写的代码，本质上就是描述单个线程要做什么。

在编程模型层面，CUDA 并 **不会直接暴露 Warp** 给你。你只会写 Thread 的代码，Warp 是硬件层面的执行方式。CUDA 源代码里不会写 `warpIdx`，但性能优化时必须考虑它（访存合并、warp divergence）。所以从**编程模型**的角度（你写代码的视角）：官方说法是 **Grid → Block → Thread**。但从“硬件执行”的角度（GPU 内部实际运行的角度）：更准确的是 **Grid → Block → Warp → Thread**。好比编程时你感觉自己写的是“一群学生各自算题”，硬件里其实是“32个学生一排，老师一次布置一道题，大家同时做”。

## 一个具体的例子

让我用个例子来说明。假设你启动了这样的配置：

```cpp
dim3 blockDim(128);   // 一个Block有128个线程
dim3 gridDim(2);      // Grid里有2个Block
```

那么实际的执行结构是这样的：

```
Grid
┌──────────────────────────────┐
│ Block(0,0)                   │
│ ┌───────────────┐            │
│ │ Warp 0        │ → Thread 0 ~ 31
│ │ Warp 1        │ → Thread 32 ~ 63
│ │ Warp 2        │ → Thread 64 ~ 95
│ │ Warp 3        │ → Thread 96 ~127
│ └───────────────┘            │
│                              │
│ Block(1,0)                   │
│ ┌───────────────┐            │
│ │ Warp 0        │ → Thread 0 ~ 31
│ │ Warp 1        │ → Thread 32 ~ 63
│ │ Warp 2        │ → Thread 64 ~ 95
│ │ Warp 3        │ → Thread 96 ~127
│ └───────────────┘            │
└──────────────────────────────┘
```

硬件执行时，每个Block内的128个线程被自动切成了128/32=4个warp，所以总共有4×2=8个warp同时在不同的SM上调度运行。

## Block和Grid的维度设计

CUDA的设计很巧妙，Grid和Block都支持最多三维。你可以这样定义一个Block：

```cpp
dim3 blockDim(x, y, z);  // x,y,z都是正整数
```

- 一维：`blockDim.x = N, blockDim.y = blockDim.z = 1`
- 二维：`blockDim.x = M, blockDim.y = N, blockDim.z = 1`  
- 三维：`blockDim.x, blockDim.y, blockDim.z`全部大于1

不过，不同架构有不同的Block上限。一般来说：
- `blockDim.x ≤ 1024`
- `blockDim.y ≤ 1024`
- `blockDim.z ≤ 64`
- 总的线程数限制：`blockDim.x × blockDim.y × blockDim.z ≤ 1024`

这意味着你可以创建一个`1024×1×1`的block（一维最大），或者`32×32×1`的block（二维，正好1024个线程），甚至`8×8×16`的block（三维，也是1024个线程）。

Grid的维度限制相对宽松一些：
- `gridDim.x ≤ 2^31-1`
- `gridDim.y ≤ 65535`
- `gridDim.z ≤ 65535`

理论上你可以启动数十亿个线程，这给了你很大的灵活性。

## 实际应用中的选择

在实战中，Block大小的选择往往遵循一些经验法则。一般来说，128或256线程的Block比较常见。而Block维度的选择更多是考虑数据结构的特性：

- 处理图像时，用二维Block（比如16×16）就很自然
- 处理体积数据时，三维Block更直观
- 简单的向量计算，一维Block就足够了

## 多维度的本质

这里要澄清一个重要的概念：Grid和Block的多维度本质上是一种语法糖。GPU的硬件执行单元（warp scheduler）根本不关心x、y、z这些维度。在硬件层面，线程只有一个全局线性ID。

所谓的`threadIdx.x`、`threadIdx.y`、`threadIdx.z`，以及`blockIdx.x`、`blockIdx.y`、`blockIdx.z`，都是编译器帮你做了线性化处理。

举个二维block的例子：

```cpp
int tid = threadIdx.x + threadIdx.y * blockDim.x;
```

三维时再加一个维度：

```cpp
int tid = threadIdx.x 
        + threadIdx.y * blockDim.x
        + threadIdx.z * blockDim.x * blockDim.y;
```

最后都变成了一个一维编号。

## 为什么提供三个维度而不是只给一维？

这是个好问题。主要有几个原因：

**可读性**：比如写图像处理时，你用`(x,y)`表示像素坐标就很直观；体数据`(x,y,z)`也一样。

**减少计算开销**：你不用手动展开4D/5D张量到1D索引再模除，而是直接拿`threadIdx.y`当行号来用。

**编程习惯**：工程师思考图像/体积/矩阵时本来就是二维三维的，这样写更符合直觉。

## 高维度数据

CUDA的Grid和Block只提供到三维的支持，那如果遇到高于三维的数据, 比如NCHW张量, 有4个维度，该怎么办呢？

> NCHW是一个在标准卷积中使用的输入数据的格式(在PyTorch, Caffe和ONNX默认的数据格式)

其实也不难，就是手动映射一下. 就像演示代码[add_bias_nchw_1d.cu](add_bias_nchw_1d.cu)里面所展示的：我们有一个4D的张量（N=2, C=3, H=4, W=5），但数据在内存中是1D连续存储的。每个线程通过自己的线程ID（`tid`）来定位要处理的元素。

```cpp
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
```

为了知道每个线程在处理4D张量的哪个位置，我们需要把1D的线程ID映射到4D坐标`(n,c,h,w)`：

```cpp
int w = tid % W;
long long t = tid / W;
int h = t % H;
t = t / H;
int c = t % C;
int n = t / C;
```

然后，为了正确访问内存，我们还需要把4D坐标转换回1D内存索引：

```cpp
long long idx = ((long long)n * C + c) * H * W + (long long)h * W + w;
```

这样就完成了完整的转换循环：1D线程ID → 4D逻辑坐标 → 1D内存索引。

下面是我运行的结果：

```bash
# 一维版本
nvcc add_bias_nchw_1d.cu -o add_bias_nchw_1d && ./add_bias_nchw_1d
out[0] = 1.5
out[1] = 2.5
out[2] = 3.5
```

从结果可以看出，程序正确地给每个输入元素都加上了1.5的偏置值，说明我们的4D到1D坐标映射和内存访问都是正确的。CUDA不提供原生4D支持，是因为3D已经能覆盖大部分物理世界的数据（向量/图像/体积），其余情况都能靠手动映射解决。

### 启动配置的语法


在 CUDA 里，“告诉编译器/运行时我要用几个维度”是在 **kernel 启动配置**时写 `<<<gridDim, blockDim>>>`。只要在 kernel 启动 `<<<...>>>` 时传入 `dim3`，编译器就知道你用几维度了。在 kernel 内部，对应的维度值会自动写进下面的`内置变量`：

* `threadIdx.x / .y / .z`
* `blockIdx.x / .y / .z`
* `blockDim.x / .y / .z`
* `gridDim.x / .y / .z`


#### 语法位置

```cpp
kernel<<< gridDim, blockDim >>>( ... );
```

`gridDim`和`blockDim`可以是整数（一维）或`dim3`结构（多维）。`dim3`是CUDA内置的一个小结构体，里面有`.x, .y, .z`三个成员，默认值都是1。

**`dim3`本身总是3维的，但你可以通过设置某些维度为1来模拟低维度的效果**。如果你写`dim3(256)`，实际上等价于一维`dim3(256, 1, 1)`；写`dim3(16, 16)`等价于二维`dim3(16, 16, 1)`；写`dim3(8, 8, 4)`就是完整的3维。没指定的维度自动设为1。

#### 具体例子

让我们继续以上面的NCHW数据（N=2, C=3, H=4, W=5）为例，看看2维和3维的block如何配置：

#### 二维block

```cpp
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
...
dim3 threadsPerBlock(16, 16);   // blockDim.x=16, blockDim.y=16
dim3 gridSize(1, 1);      // 因为H=4<16, W=5<16
kernel<<<gridSize, threadsPerBlock>>>(...);
```

这样每个block是16×16=256个线程，grid是1×1=1个block。
`threadIdx.x`直接对应W维度，`threadIdx.y`直接对应H维度，但仍需要循环遍历N和C。

#### 三维block

```cpp
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
...
dim3 threadsPerBlock(8, 8, 4);   // blockDim=(8,8,4)=256
dim3 gridSize(1, 1, 1);    // 因为W=5<8, H=4<8, C=3<4
kernel<<<gridSize, threadsPerBlock>>>(...);
```

每个block有8×8×4=256个线程，grid是1×1×1=1个block。 `threadIdx.x`对应W，`threadIdx.y`对应H，`threadIdx.z`对应C，只需要循环遍历N维度。

我在`add_bias_nchw_1d.cu`、`add_bias_nchw_2d.cu`和`add_bias_nchw_3d.cu`三个文件中分别实现了这三种方法。

编译和运行命令：
```bash
# 二维版本  
nvcc add_bias_nchw_2d.cu -o add_bias_nchw_2d && ./add_bias_nchw_2d
# 三维版本
nvcc add_bias_nchw_3d.cu -o add_bias_nchw_3d && ./add_bias_nchw_3d
```

虽然三种方法的block配置不同，但都会得到相同的结果：每个输入元素都加上了1.5的偏置值。

## Block中的Warp划分

现在让我们深入看看Warp是如何在Block中划分的。这很重要，因为它直接影响性能。

假设我们有一个**二维Block（blockDim = 10 x 10 = 100个线程）**，现在看看warp是怎么在二维线程里切分的。

### 二维Block的布局

Block内部逻辑上是一个10×10的二维线程网格：

```
ThreadIdx
( x , y )

y=0 → (0,0) (1,0) (2,0) ... (9,0)
y=1 → (0,1) (1,1) (2,1) ... (9,1)
...
y=9 → (0,9) (1,9) (2,9) ... (9,9)
```

总共有100个线程。

### Warp的划分方式

这里有个关键点：硬件不会关心二维，而是把线程按**行优先**线性展开（row-major顺序：先x，再y）：

```
Thread Linear ID = y * blockDim.x + x
```

所以实际的warp划分是：
- 第0个Warp → 线程0~31
- 第1个Warp → 线程32~63  
- 第2个Warp → 线程64~95
- 第3个Warp → 线程96~99（只有4个线程，其余28个"空位"浪费掉了）

### 可视化图示

```
Block(10 x 10)
y=0: [00][01][02][03][04][05][06][07][08][09]
y=1: [10][11][12][13][14][15][16][17][18][19]
y=2: [20][21][22][23][24][25][26][27][28][29]
y=3: [30][31][32][33][34][35][36][37][38][39]
y=4: [40][41][42][43][44][45][46][47][48][49]
y=5: [50][51][52][53][54][55][56][57][58][59]
y=6: [60][61][62][63][64][65][66][67][68][69]
y=7: [70][71][72][73][74][75][76][77][78][79]
y=8: [80][81][82][83][84][85][86][87][88][89]
y=9: [90][91][92][93][94][95][96][97][98][99]

Warp划分：
Warp 0 → [00]~[31]
Warp 1 → [32]~[63]
Warp 2 → [64]~[95]
Warp 3 → [96]~[99] (不满32,有空线程)
```

硬件以32个线程为一组（warp）执行。如果线程数不是32的倍数，最后那个不满32的warp里有些"空座位"，这些执行槽位也会被占用但不干活，效率会下降。

## CUDA的编程模型总结

让我总结一下这个小节的内容：

1. **CUDA的编程模型**：Grid → Block → Thread，但硬件执行时是Grid → Block → Warp → Thread
2. **维度设计**：Block和Grid都支持最多三维，这主要是为了代码的可读性和编程便利性. 其**本质**只是语法糖，硬件层面都是线性化的线程ID
3. **Warp划分**：硬件按线性ID来分warp，即使是二维/三维block，warp仍然是32个线程为一组
4. **实际应用**：选择Block大小时要考虑warp对齐，选择维度时要考虑数据结构的自然表达

理解这些概念对于写出高效的CUDA代码很重要。虽然看起来有点复杂，但一旦理解了，你就能更好地控制GPU的执行方式，避免一些常见的性能陷阱。




