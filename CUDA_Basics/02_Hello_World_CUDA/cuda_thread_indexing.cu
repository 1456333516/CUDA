#include <stdio.h>

// 定义一个CUDA内核函数，该函数将在GPU上执行
// 每个线程会打印出它所属的block ID和thread ID、
// gridDin.x - 配置grid_size的数值
// blockDin.x - 配置block_size的数值    
__global__ void hello_from_gpu() {
    // 获取当前线程所在的block ID
    const int bid = blockIdx.x;
    // 获取当前线程在block中的thread ID
    const int tid = threadIdx.x;
    // 打印Hello World信息，包含block ID和thread ID
    printf("Hello World from block %d and thread %d!!\n", bid, tid);
}

int main() {
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}