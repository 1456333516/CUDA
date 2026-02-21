#include <stdio.h>

// 定义一个CUDA内核函数，该函数将在GPU上执行
__global__ void hello_from_gpu() {
    // 在GPU上打印"Hello World from GPU!"
    // 注意：这里使用printf而不是cout，因为CUDA内核函数运行在GPU上，
    // 而cout是C++标准库的一部分，只能在CPU上使用。
    printf("Hello World from GPU!\n");
}

int main() {
    // 启动CUDA内核函数，配置为1个block，每个block有1个thread
    hello_from_gpu<<<1,1>>>();
    
    // 等待GPU上的所有任务完成，确保输出能够显示
    cudaDeviceSynchronize();
    
    return 0;
}