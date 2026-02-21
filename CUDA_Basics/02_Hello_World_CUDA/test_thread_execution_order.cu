#include <stdio.h>

// 定义一个CUDA内核函数，该函数将在GPU上执行
// 每个线程会打印出它所属的block ID和thread ID（二维）
__global__ void hello_from_gpu() {
    // 获取当前线程所在的block ID（一维）
    const int b = blockIdx.x;
    // 获取当前线程在block中的thread ID（x方向）
    const int tx = threadIdx.x;
    // 获取当前线程在block中的thread ID（y方向）
    const int ty = threadIdx.y;
    // 打印Hello World信息，包含block ID和thread ID（二维坐标）
    printf("Hello World from block-%d and thread-(%d,%d)!\n", b, tx, ty);
}

int main() 
{
    // 定义block的大小为2x4（即每个block有2行4列的线程）
    const dim3 block_size(2, 4);
    // 配置grid_size 也是这样 const dim3 grid_size(2,4);
    // block_size和grid_size都有3个维度，包括x、y和z三个维度。像上面那样配置，x为2，y为4，z为1。没有配置的维度默认为1。
    // 启动CUDA内核函数，配置为1个block，每个block的大小为block_size
    // x的线程指标threadIdx.x是最内层（变化最快）
    hello_from_gpu<<<1, block_size>>>();
    // 等待所有GPU操作完成，确保输出信息被打印
    cudaDeviceSynchronize();
    return 0;
}