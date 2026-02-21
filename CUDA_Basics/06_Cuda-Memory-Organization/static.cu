#include "error.cuh"  // 包含CUDA错误检查宏
#include <stdio.h>    // 包含标准输入输出库

// 声明设备端全局变量（驻留在GPU全局内存中）
__device__ int d_x = 1;     // 设备端标量变量，初始化为1
__device__ int d_y[2];      // 设备端数组变量，大小为2个整数

// CUDA内核函数：在GPU上执行的并行函数
void __global__ my_kernel(void)
{
    // 对设备端数组d_y的两个元素分别加上d_x的值
    d_y[0] += d_x;  // d_y[0] = d_y[0] + d_x
    d_y[1] += d_x;  // d_y[1] = d_y[1] + d_x
    
    // 打印当前设备端变量的值（注意：每个线程都会执行此打印）
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d\n", d_x, d_y[0], d_y[1]);
}

// 主函数：主机端程序入口
int main(void)
{
    // 声明并初始化主机端数组
    int h_y[2] = {10, 20};  // 主机端数组，初始值为{10, 20}
    
    // 将主机端数据复制到设备端符号内存
    // cudaMemcpyToSymbol用于将主机数据复制到已声明的设备变量
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));

    // 启动CUDA内核函数，配置为1个block，1个thread
    my_kernel<<<1, 1>>>();
    
    // 等待GPU内核执行完成，确保所有操作都已完成
    CHECK(cudaDeviceSynchronize());

    // 将设备端数据复制回主机端
    // cudaMemcpyFromSymbol用于从设备变量复制数据到主机
    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    
    // 打印主机端数组的最终值
    printf("h_y[0] = %d, h_y[1] = %d\n", h_y[0], h_y[1]);

    return 0;  // 程序正常退出
}