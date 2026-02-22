#include "error.cuh"  // 包含自定义的CUDA错误检查宏CHECK

/*
 * ==================== 数据类型配置 ====================
 * 条件编译：这是一个重要的编程技巧，让我们可以根据需要选择不同的数据精度
 * 编译选项说明：
 * - 不加任何参数编译：使用单精度float（默认）
 * - 使用-DUSE_DP参数编译：使用双精度double
 *   例如：nvcc -DUSE_DP reduce.cu -o reduce_double
 */

#ifdef USE_DP
    // 如果定义了USE_DP宏，就使用双精度浮点数
    typedef double real;           // double类型占用64位，精度更高但速度稍慢
    const real EPSILON = 1.0e-15;  // 双精度下的极小值，用于比较两个浮点数是否相等
#else
    // 如果没有定义USE_DP宏，就使用单精度浮点数（默认情况）
    typedef float real;            // float类型占用32位，速度快但精度较低
    const real EPSILON = 1.0e-6;   // 单精度下的极小值，用于浮点数比较
#endif

/*
 * ==================== 常量定义 ====================
 */
#define N 1024                    // 要处理的数据元素总数
#define BLOCK_SIZE 256            // 每个线程块中的线程数量
                                  // 选择256是因为它通常是GPU的最佳实践值

/*
 * ==================== CUDA核函数定义 ====================
 * CUDA核函数：执行归约操作（求和）
 * 归约操作是将大量数据通过某种运算（如加法）合并成一个单一结果的过程
 * 例如：[1, 2, 3, 4, 5] 经过加法归约后得到 15
 */
__global__ void reduce_kernel(const real *d_input, real *d_output, int n) {
    /*
     * 共享内存声明：用于线程块内线程间的数据交换
     * extern关键字表示这个数组的大小在启动核函数时动态指定
     * 这样可以灵活地根据BLOCK_SIZE调整共享内存大小
     */
    extern __shared__ real sdata[];
    
    /*
     * 线程索引计算：
     * - threadIdx.x: 当前线程在块内的索引 (0到BLOCK_SIZE-1)
     * - blockIdx.x: 当前块的索引
     * - blockDim.x: 每个块的线程数量 (BLOCK_SIZE)
     * - idx: 当前线程处理的全局数据索引
     */
    int tid = threadIdx.x;        // 线程在块内的编号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局数据索引
    
    /*
     * 数据加载阶段：将全局内存中的数据加载到共享内存
     * 边界检查：如果索引超出数据范围，则使用0填充
     * 这样可以处理数据量不是BLOCK_SIZE整数倍的情况
     */
    sdata[tid] = (idx < n) ? d_input[idx] : 0;
    
    /*
     * 线程同步点1：确保所有线程都完成了数据加载
     * __syncthreads()是一个屏障，让同一块内的所有线程在此等待
     * 直到所有线程都到达这个点才继续执行
     */
    __syncthreads();
    
    /*
     * 树形归约算法：逐步将数据两两相加
     * 这是一种高效的并行归约方法，时间复杂度为O(log n)
     * 
     * 执行过程可视化（以BLOCK_SIZE=8为例）：
     * 初始状态: [1][2][3][4][5][6][7][8]
     * 第1轮:    [3][7][11][15][0][0][0][0]  (相邻元素相加)
     * 第2轮:    [10][26][0][0][0][0][0][0]  (间隔2个位置相加)
     * 第3轮:    [36][0][0][0][0][0][0][0]   (间隔4个位置相加)
     * 最终结果: 36 存储在线程0中
     */
    
    // 步骤1：每隔2个元素进行一次相加（处理128对元素）
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();  // 同步确保所有线程完成本轮计算
    
    // 步骤2：每隔4个元素进行一次相加（处理64对元素）
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();  // 同步
    
    /*
     * 步骤3：Warp级别优化
     * Warp是GPU的基本执行单元，通常包含32个线程
     * 同一warp内的线程执行相同指令，不需要__syncthreads()同步
     * 这种优化可以减少同步开销，提高性能
     */
    if (tid < 32) {
        // 无需__syncthreads()，因为都在同一个warp中
        sdata[tid] += sdata[tid + 32];  // 处理32对元素
        sdata[tid] += sdata[tid + 16];  // 处理16对元素
        sdata[tid] += sdata[tid + 8];   // 处理8对元素
        sdata[tid] += sdata[tid + 4];   // 处理4对元素
        sdata[tid] += sdata[tid + 2];   // 处理2对元素
        sdata[tid] += sdata[tid + 1];   // 处理1对元素
    }
    
    /*
     * 结果写回：只有第一个线程负责将部分结果写回全局内存
     * 使用原子操作atomicAdd确保多个线程块的结果能正确累加
     * 原子操作的重要性：防止多个线程同时写入时产生竞争条件
     */
    if (tid == 0) {
        atomicAdd(d_output, sdata[0]);  // 原子加法：线程安全的累加操作
    }
}

/*
 * ==================== 主机函数定义 ====================
 * 主机函数：执行GPU上的归约操作
 * 这是CPU端调用的函数，负责管理GPU资源和数据传输
 */
real reduce(const real *d_z)  // d_z: 指向GPU显存中输入数据的指针
{
    /*
     * 网格配置计算：确定需要多少个线程块
     * 公式解释：(N + BLOCK_SIZE - 1) / BLOCK_SIZE 实现向上取整
     * 例如：N=1024, BLOCK_SIZE=256 → 需要4个线程块
     *      N=1025, BLOCK_SIZE=256 → 需要5个线程块
     */
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    /*
     * 共享内存大小计算：每个线程块需要的共享内存字节数
     * 每个线程需要一个real类型的空间来存储临时数据
     * 例如：BLOCK_SIZE=256, real=float(4字节) → smem=1024字节
     */
    const int smem = sizeof(real) * BLOCK_SIZE;

    /*
     * 主机端结果变量：在CPU内存中存储最终结果
     * 初始化为0，作为累加的起点
     */
    real h_y[1] = {0};  // 数组形式便于后续内存拷贝操作
    
    /*
     * GPU端指针声明：指向GPU显存中结果存储位置
     */
    real *d_y;  // d_前缀表示device(GPU)端变量
    
    /*
     * GPU内存分配：为结果变量在GPU显存中分配空间
     * 注意：原代码中的cudaMaloc应该是cudaMalloc的拼写错误
     * CHECK宏会自动检查分配是否成功，失败时会打印详细错误信息并退出程序
     */
    CHECK(cudaMalloc(&d_y, sizeof(real)));  // 分配一个real大小的GPU内存
    
    /*
     * 数据初始化：将主机端的初始值0复制到GPU显存中
     * 这很重要，因为GPU显存中的内容是未初始化的随机值
     * cudaMemcpy参数说明：
     * - 目标地址：d_y (GPU端)
     * - 源地址：h_y (CPU端)  
     * - 数据大小：sizeof(real)
     * - 传输方向：cudaMemcpyHostToDevice (从主机到设备)
     */
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    /*
     * CUDA核函数启动：在GPU上执行归约操作
     * 核函数启动语法：kernel_name<<<grid_size, block_size, shared_mem>>>(parameters...)
     * 参数详解：
     * - grid_size: 线程块的数量 (4个块)
     * - BLOCK_SIZE: 每个线程块中的线程数量 (256个线程/块)
     * - smem: 每个线程块分配的共享内存大小 (1024字节)
     * - d_z: 输入数据指针
     * - d_y: 输出结果指针
     * - N: 数据元素总数
     */
    reduce_kernel<<<grid_size, BLOCK_SIZE, smem>>>(d_z, d_y, N);
    
    /*
     * 错误检查：检查CUDA核函数执行过程中是否出错
     * 由于核函数是异步执行的，错误可能在稍后才被发现
     * 这两个检查对于调试非常重要
     */
    CHECK(cudaGetLastError());      // 检查核函数启动是否出错
    CHECK(cudaDeviceSynchronize()); // 等待所有GPU操作完成，确保能捕获运行时错误
    
    /*
     * 结果获取：将GPU计算结果复制回主机内存
     * 传输方向：cudaMemcpyDeviceToHost (从设备到主机)
     */
    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    
    /*
     * 资源清理：释放GPU显存，避免内存泄漏
     * 每次cudaMalloc都应该对应一次cudaFree
     */
    CHECK(cudaFree(d_y));

    /*
     * 返回最终结果：h_y[0]包含了所有线程块归约后的总和
     */
    return h_y[0];
}