#include <math.h>  // 包含数学函数库
#include <stdio.h> // 包含标准输入输出库

// 定义一个很小的浮点数，用于比较两个浮点数是否相等
const double EPSILON = 1.0e-15;
// 定义常量a和b，作为数组x和y的初始值
const double a = 1.23;
const double b = 2.34;
// 定义常量c，作为期望的结果值（a + b）
const double c = 3.57;

// 声明CUDA内核函数add，用于在GPU上并行执行数组加法
__global__ void add(const double *x, const double *y, double *z);
// 声明check函数，用于在CPU上检查数组z中的值是否等于期望值c
void check(const double *z, const int N);

int main(void)
{
    // 定义数组大小N
    const int N = 100000000;
    // 计算数组所需的内存大小M（以字节为单位）
    const int M = sizeof(double) * N;

    // 在主机（CPU）上动态分配内存给数组h_x、h_y和h_z
    double *h_x = (double *)malloc(M);
    double *h_y = (double *)malloc(M);
    double *h_z = (double *)malloc(M);

    // 初始化主机上的数组h_x和h_y，所有元素都设置为常量a和b
    for (int n = 0; n < N; n++)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    // 声明设备（GPU）上的指针
    double *d_x, *d_y, *d_z;
    // 在设备上分配内存给数组d_x、d_y和d_z
    // 指针的指针，cudaMalloc 需要知道你想要分配的内存地址要保存在哪里。d_x 是一个指针变量，用来记录 GPU 分配的内存地址。
    // 但是 cudaMalloc 不能直接修改 d_x 的值，因为它只能通过“指针的指针”来修改原始变量。
    // cudaMalloc 函数原型：
    // cudaMalloc(void **devPtr, size_t size);
    //
    // 参数说明：
    // 1. void **devPtr:
    //    - 这是一个指向设备（GPU）指针的指针。
    //    - 用于接收 cudaMalloc 在 GPU 显存中分配的内存地址。
    //    - 示例：&d_x 表示将分配的地址保存到 d_x 中。
    //
    // 2. size_t size:
    //    - 这是要分配的内存大小（以字节为单位）。
    //    - 示例：M 表示分配 M 字节的内存。
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);

    // 将主机上的数据复制到设备上
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    // 定义每个block的线程数和grid中的block数
    const int block_size = 128;
    const int grid_size = N / block_size;

    // 启动CUDA内核函数add，在GPU上并行执行数组加法
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    // 将设备上的结果复制回主机
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);

    // 调用check函数，验证主机上的数组h_z中的值是否正确
    check(h_z, N);

    // 释放主机上的内存
    free(h_x);
    free(h_y);
    free(h_z);

    // 释放设备上的内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}

// CUDA内核函数add的实现：将数组x和y逐元素相加，结果存储在数组z中
__global__ void add(const double *x, const double *y, double *z)
{
    // 计算当前线程的全局索引
    // 让每个线程知道它应该处理数组中的哪一个元素。
    // 这是每个 block 中的线程数量（沿 x 方向）。比如你设置了 block_size = 128，那么 blockDim.x = 128。其它同理。
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    // 执行加法操作
    if(n < N)
    {
        z[n] = x[n] + y[n];
    }
    // 线程数量大于数组长度，则返回.可以使用return，核函数没有返回值。和函数只能用void，无返回值
    if (n >= N)
        return;
}