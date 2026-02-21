#include "error.cuh"  // 包含CUDA错误检查宏
#include <stdio.h>    // 包含标准输入输出库

// 主函数：查询并显示CUDA设备属性
int main(int argc, char *argv[])
{
    // 初始化设备ID，默认为0
    int device_id = 0;
    
    // 如果命令行参数存在，则使用第一个参数作为设备ID
    if(argc > 1)
        device_id = atoi(argv[1]);  // 将字符串参数转换为整数
    
    // 设置当前使用的CUDA设备
    CHECK(cudaSetDevice(device_id));

    // 声明设备属性结构体变量
    cudaDeviceProp prop;
    
    // 获取指定设备的属性信息
    CHECK(cudaGetDeviceProperties(&prop, device_id));

    // 打印设备基本信息
    printf("device_id: %d\n", device_id);                    // 设备ID
    printf("device name: %s\n", prop.name);                  // 设备名称
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);  // 计算能力版本

    // 打印内存相关信息
    printf("Amount of global memory: %g GB\n",               // 全局内存总量
           prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Amount of constant memory: %g KB\n",             // 常量内存总量
           prop.totalConstMem / 1024.0);

    // 打印网格和线程块的最大尺寸限制
    printf("Maximum grid size: %d %d %d\n",                  // 最大网格维度
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size: %d %d %d\n",                 // 最大线程块维度
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

    // 打印流多处理器(SM)相关信息
    printf("Number of SMs: %d\n", prop.multiProcessorCount); // SM数量

    // 打印共享内存信息
    printf("Maximum amount of shared memory per block: %g KB\n",  // 每块最大共享内存
           prop.sharedMemPerBlock / 1024.0);
    printf("Maximum amount of shared memory per SM: %g KB\n",     // 每SM最大共享内存
           prop.sharedMemPerMultiprocessor / 1024.0);

    // 打印寄存器信息
    printf("Maximum number of registers per block: %d K\n",       // 每块最大寄存器数
           prop.regsPerBlock / 1024);
    printf("Maximum number of registers per SM: %d K\n",          // 每SM最大寄存器数
           prop.regsPerMultiprocessor / 1024);

    // 打印线程数量限制
    printf("Maximum number of threads per block: %d\n",           // 每块最大线程数
           prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM: %d\n",              // 每SM最大线程数
           prop.maxThreadsPerMultiProcessor);

    return 0;  // 程序正常退出
}
