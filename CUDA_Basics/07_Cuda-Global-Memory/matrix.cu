// 条件编译：根据是否定义USE_DP来选择数据精度
#ifdef USE_DP
    typedef double real;           // 使用双精度浮点数(64位)
    const real EPSILON = 1.0e-15;  // 双精度下的极小值常量，用于数值比较
#else
    typedef float real;            // 使用单精度浮点数(32位)
    const real EPSILON = 1.0e-6;   // 单精度下的极小值常量，用于数值比较
#endif
const int TILE_DIM = 32;//C++风格
// 等价于#define TILE_DIM 32 //C风格
const real x0 = 100.0;  // 定义目标值常量，用于后续可能的平方根比较操作

// CUDA内核函数：二维矩阵复制操作
// 参数说明：
// - A: 源矩阵指针(只读)
// - B: 目标矩阵指针(可写)
// - N: 矩阵的维度大小(N×N)
void __global__ copy(const real *A, real *B, int N)
{
    // 计算当前线程在矩阵中的x坐标(列索引)
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    
    // 计算当前线程在矩阵中的y坐标(行索引)
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 将二维坐标转换为一维索引(行优先存储)
    const int index = ny * N + nx;
    
    // 边界检查：确保线程处理的元素在矩阵范围内
    if(nx < N && ny < N)
    {
        // 执行矩阵元素复制操作：B[index] = A[index]
        B[index] = A[index];
    }
}

// 网格配置参数计算
const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;  // x方向的网格块数量(向上取整)
const int grid_size_y = grid_size_x;                     // y方向的网格块数量(假设方形矩阵)

// 定义线程块的维度配置
const dim3 block_size(TILE_DIM, TILE_DIM);               // 每个线程块包含TILE_DIM×TILE_DIM个线程

// 定义网格的维度配置
const dim3 grid_size(grid_size_x, grid_size_y);          // 网格包含grid_size_x×grid_size_y个线程块

// 启动CUDA内核函数进行矩阵复制
// 参数说明：
// - d_A: 设备端源矩阵指针
// - d_B: 设备端目标矩阵指针
// - N: 矩阵维度
copy<<<grid_size, block_size>>>(d_A, d_B, N);