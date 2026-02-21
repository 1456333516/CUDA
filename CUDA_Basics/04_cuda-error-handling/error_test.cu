#include "error.cuh" // 包含自定义的 CUDA 错误检查宏 CHECK
#include <math.h>    // 包含数学函数库，用于浮点数比较等操作
#include <stdio.h>   // 包含标准输入输出库，用于打印调试信息

// 定义一个很小的浮点数，用于比较两个浮点数是否相等（容差）
const double EPSILON = 1.0e-15;

// 定义常量 a 和 b，作为数组 x 和 y 的初始值
const double a = 1.23;
const double b = 2.34;

// 定义常量 c，作为期望的结果值（a + b = 3.57）
const double c = 3.57;

// 声明 CUDA 内核函数 add，用于在 GPU 上并行执行数组加法
__global__ void add(const double *x, const double *y, double *z);

// 声明 check 函数，用于在 CPU 上检查数组 z 中的值是否等于期望值 c
void check(const double *z, const int N);

int main(void)
{
    // 定义数组大小 N（1亿个元素）
    const int N = 100000000;

    // 计算数组所需的内存大小 M（以字节为单位）
    const int M = sizeof(double) * N;

    // 在主机（CPU）上动态分配内存给数组 h_x、h_y 和 h_z
    double *h_x = (double *)malloc(M);
    double *h_y = (double *)malloc(M);
    double *h_z = (double *)malloc(M);

    // 初始化主机上的数组 h_x 和 h_y，所有元素都设置为常量 a 和 b
    for (int n = 0; n < N; n++)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    // 声明设备（GPU）上的指针
    double *d_x, *d_y, *d_z;

    // 在 GPU 上分配内存，并使用 CHECK 宏检查是否成功
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMalloc((void **)&d_y, M));
    CHECK(cudaMalloc((void **)&d_z, M));

    // 将主机上的数据复制到设备上（从主机到设备）
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    // 定义每个 block 的线程数和 grid 中的 block 数
    const int block_size = 128;           // 每个 block 包含 128 个线程
    const int grid_size = N / block_size; // 计算需要多少个 block 来处理所有数据

    // 启动 CUDA 内核函数 add，在 GPU 上并行执行数组加法
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    // 检查内核启动是否有错误
    CHECK(cudaGetLastError());

    // 等待 CUDA 内核函数执行完毕
    CHECK(cudaDeviceSynchronize());

    // 将设备上的结果复制回主机（从设备到主机）
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);

    // 调用 check 函数，验证主机上的数组 h_z 中的值是否正确
    check(h_z, N);

    // 释放主机上的内存
    free(h_x);
    free(h_y);
    free(h_z);

    // 释放设备上的内存，并使用 CHECK 宏检查是否成功
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));

    return 0; // 程序正常结束
}

// CUDA 内核函数 add 的实现：将数组 x 和 y 逐元素相加，结果存储在数组 z 中
__global__ void add(const double *x, const double *y, double *z)
{
    // 计算当前线程的全局索引
    // 让每个线程知道它应该处理数组中的哪一个元素。
    // blockDim.x 是每个 block 中的线程数量（沿 x 方向）。
    // blockIdx.x 是当前 block 的索引，threadIdx.x 是当前线程在 block 中的索引。
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    // 执行加法操作：z[n] = x[n] + y[n]
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }

    // 如果线程索引超出数组范围，则直接返回（可选优化）
    if (n >= N)
        return;
}