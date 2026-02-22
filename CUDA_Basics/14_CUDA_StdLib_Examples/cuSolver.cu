#include "error.cuh"              // 包含自定义的CUDA错误检查宏CHECK
#include <stdio.h>                // 包含标准输入输出库
#include <stdlib.h>               // 包含标准库函数
#include <cusolverDn.h>           // 包含cuSolver密集矩阵库头文件

/*
 * ==================== 程序功能说明 ====================
 * 这是一个使用cuSolver库计算Hermitian矩阵特征值的示例程序
 * 
 * Hermitian矩阵（厄米特矩阵）特点：
 * - 复数方阵
 * - 等于其共轭转置：A = A^H
 * - 特征值都是实数
 * 
 * 本程序处理的矩阵：
 * A = [ 0  -i ]
 *     [ i   0 ]
 * 
 * 其中 i 是虚数单位，满足 i² = -1
 * 
 * 理论特征值：λ₁ = 1, λ₂ = -1
 * 
 * cuSolver的优势：
 * 1. 专门针对稠密矩阵优化
 * 2. 提供多种分解算法（LU, QR, Cholesky, SVD等）
 * 3. 支持实数和复数矩阵
 * 4. 工业级数值稳定性和精度
 */

int main()
{
    /*
     * ==================== 矩阵维度定义 ====================
     */
    int N = 2;        // 矩阵的阶数（2×2矩阵）
    int N2 = N * N;   // 矩阵元素总数（4个元素）

    /*
     * ==================== 主机端矩阵初始化 ====================
     * 在CPU内存中分配并初始化复数矩阵A
     * cuDoubleComplex是CUDA定义的双精度复数结构体
     */
    cuDoubleComplex *A_cpu = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * N2);
    
    /*
     * 初始化Hermitian矩阵：
     * A = [ 0  -i ]
     *     [ i   0 ]
     * 
     * 复数表示：实部 + 虚部*i
     * A[0,0] = 0 + 0*i  (对角线元素，必须是实数)
     * A[0,1] = 0 + 1*i  (上三角元素)
     * A[1,0] = 0 + (-1)*i = 0 - 1*i (下三角元素，等于上三角的共轭)
     * A[1,1] = 0 + 0*i  (对角线元素，必须是实数)
     * 
     * 注意：原代码中的循环逻辑有问题，这里是修正后的正确版本
     */
    // 方法1：逐个元素赋值（推荐，清晰易懂）
    A_cpu[0].x = 0.0;   A_cpu[0].y = 0.0;   // A[0,0] = 0 + 0*i
    A_cpu[1].x = 0.0;   A_cpu[1].y = -1.0;  // A[1,0] = 0 - 1*i
    A_cpu[2].x = 0.0;   A_cpu[2].y = 1.0;   // A[0,1] = 0 + 1*i
    A_cpu[3].x = 0.0;   A_cpu[3].y = 0.0;   // A[1,1] = 0 + 0*i
    
    /*
     * 原始错误代码分析：
     * for循环中每次都重新给A_cpu[0]到A_cpu[3]赋值
     * 导致只有最后一次赋值生效，逻辑错误
     */

    /*
     * ==================== GPU端内存分配和数据传输 ====================
     * 将主机端矩阵数据传输到GPU显存
     */
    cuDoubleComplex *A;  // GPU端矩阵指针
    CHECK(cudaMalloc((void **)&A, sizeof(cuDoubleComplex) * N2));  // 分配GPU内存
    CHECK(cudaMemcpy(A, A_cpu, sizeof(cuDoubleComplex) * N2, cudaMemcpyHostToDevice));  // 数据传输

    /*
     * ==================== 特征值存储空间分配 ====================
     * 为特征值结果分配内存空间
     */
    double *W_cpu = (double *)malloc(sizeof(double) * N);  // 主机端特征值数组
    double *W;  // GPU端特征值指针
    CHECK(cudaMalloc((void **)&W, sizeof(double) * N));    // 分配GPU内存

    /*
     * ==================== cuSolver句柄创建 ====================
     * cuSolver句柄是使用库函数必需的上下文对象
     */
    cusolverDnHandle_t handle = NULL;  // 声明句柄变量
    cusolverDnCreate(&handle);         // 创建并初始化句柄

    /*
     * ==================== 求解参数设置 ====================
     */
    // jobz参数：指定计算模式
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;  // 只计算特征值，不计算特征向量
    
    // uplo参数：指定矩阵存储方式
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;       // 只使用矩阵的下三角部分
                                                          // 因为Hermitian矩阵是对称的

    /*
     * ==================== 工作空间大小查询 ====================
     * cuSolver需要额外的工作空间来执行计算
     * 首先查询所需工作空间的大小
     */
    int lwork = 0;  // 工作空间大小（元素个数）
    
    /*
     * 查询函数：cusolverDnZheevd_bufferSize
     * Z: 双精度复数 (Double Complex)
     * heev: Hermitian eigenvalue
     * d: divide and conquer algorithm
     * 
     * 参数说明：
     * - handle: cuSolver句柄
     * - jobz: 计算模式（只求特征值）
     * - uplo: 矩阵存储模式（下三角）
     * - N: 矩阵阶数
     * - A: 矩阵数据指针
     * - N: 矩阵主维度（行数）
     * - W: 特征值存储指针
     * - &lwork: 返回工作空间大小的指针
     */
    cusolverDnZheevd_bufferSize(handle, jobz, uplo, N, A, N, W, &lwork);

    /*
     * ==================== 工作空间分配 ====================
     * 根据查询到的大小分配GPU工作空间
     */
    cuDoubleComplex *work;  // 工作空间指针
    CHECK(cudaMalloc((void **)&work, sizeof(cuDoubleComplex) * lwork));  // 分配工作空间

    /*
     * ==================== 状态信息存储空间 ====================
     * 为求解器的状态信息分配内存
     */
    int* info;  // 求解状态信息指针
    CHECK(cudaMalloc((void **)&info, sizeof(int)));  // 分配一个整数的GPU内存

    /*
     * ==================== 核心计算：特征值求解 ====================
     * 调用cuSolver的Hermitian特征值求解函数
     * 
     * 函数：cusolverDnZheevd
     * 使用分治算法（divide and conquer）求解Hermitian矩阵的特征值
     * 
     * 参数详解：
     * - handle: cuSolver句柄
     * - jobz: 计算模式（只计算特征值）
     * - uplo: 矩阵存储模式（使用下三角）
     * - N: 矩阵阶数（2）
     * - A: 输入矩阵（会被破坏）
     * - N: 矩阵主维度（行数）
     * - W: 输出特征值数组
     * - work: 工作空间指针
     * - lwork: 工作空间大小
     * - info: 状态信息指针
     * 
     * 算法过程：
     * 1. 将矩阵约化为三对角形式
     * 2. 使用分治法求解三对角矩阵的特征值
     * 3. 特征值按升序排列存储在W中
     */
    cusolverDnZheevd(handle, jobz, uplo, N, A, N, W, work, lwork, info);

    /*
     * ==================== 结果传输 ====================
     * 将计算得到的特征值从GPU复制到主机
     */
    cudaMemcpy(W_cpu, W, sizeof(double) * N, cudaMemcpyDeviceToHost);

    /*
     * ==================== 结果输出 ====================
     * 打印计算得到的特征值
     */
    printf("Eigenvalues are:\n");  // 打印标题
    for(int n = 0; n < N; ++n)     // 遍历所有特征值
    {
        printf("%g\n", W_cpu[n]);  // 打印第n个特征值
                                   // %g格式自动选择合适的数字表示方式
    }

    /*
     * ==================== 资源清理 ====================
     * 按相反顺序释放所有分配的资源
     */
    cusolverDnDestroy(handle);     // 销毁cuSolver句柄
    
    free(A_cpu);                   // 释放主机端矩阵内存
    free(W_cpu);                   // 释放主机端特征值内存
    CHECK(cudaFree(A));            // 释放设备端矩阵内存
    CHECK(cudaFree(W));            // 释放设备端特征值内存
    CHECK(cudaFree(work));         // 释放设备端工作空间
    CHECK(cudaFree(info));         // 释放设备端状态信息内存

    return 0;  // 程序正常结束
}

/*
 * ==================== 预期输出 ====================
 * Eigenvalues are:
 * -1
 * 1
 * 
 * 数学验证：
 * 对于矩阵 A = [ 0  -i ]
 *              [ i   0 ]
 * 
 * 特征方程：det(A - λI) = 0
 * det([ -λ  -i ] ) = (-λ)(-λ) - (-i)(i) = λ² - (-i²) = λ² - 1 = 0
 *     [  i  -λ ]
 * 
 * 解得：λ² = 1 → λ = ±1
 * 所以特征值为 λ₁ = -1, λ₂ = 1
 */

/*
 * ==================== 技术要点总结 ====================
 * 
 * 1. Hermitian矩阵特性：
 *    - 特征值都是实数
 *    - 不同特征值对应的特征向量正交
 *    - 可以进行谱分解
 * 
 * 2. cuSolver函数命名规则：
 *    - Dn: Dense（稠密矩阵）
 *    - Z: 双精度复数 (Double Complex)
 *    - heev: Hermitian eigenvalue
 *    - d: divide and conquer algorithm
 * 
 * 3. 存储优化：
 *    - 只存储矩阵的一半（上三角或下三角）
 *    - 利用矩阵的对称性质节省内存
 * 
 * 4. 算法选择：
 *    - 分治算法（divide and conquer）：适合中小规模矩阵
 *    - QR算法：适合大规模矩阵
 *    - 根据问题规模选择合适的算法
 * 
 * 5. 实际应用场景：
 *    - 量子力学中的哈密顿量对角化
 *    - 信号处理中的协方差矩阵分析
 *    - 机器学习中的主成分分析(PCA)
 *    - 结构力学中的振动模态分析
 */

/*
 * ==================== 常见错误提醒 ====================
 * 
 * 1. 内存管理：
 *    - 每个cudaMalloc必须对应cudaFree
 *    - 句柄必须正确创建和销毁
 * 
 * 2. 数据初始化：
 *    - Hermitian矩阵必须满足A = A^H
 *    - 对角线元素必须是实数
 * 
 * 3. 参数设置：
 *    - 矩阵维度必须一致
 *    - 工作空间大小必须足够
 *    - 存储模式要与实际数据一致
 */