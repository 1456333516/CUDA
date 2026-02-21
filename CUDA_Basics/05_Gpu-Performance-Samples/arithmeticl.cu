#include <math.h>  // 包含数学函数库，用于sqrt()等数学运算

// 条件编译：根据是否定义USE_DP来选择数据精度
// 双精度运算：nvcc -O3 -arch=sm_75 -DUSE_DP arithmetic.cu
// 单精度运算：nvcc -O3 -arch=sm_75 arithmetic.cu
#ifdef USE_DP
    typedef double real;           // 使用双精度浮点数
    const real EPSILON = 1.0e-15;  // 双精度下的极小值常量
#else
    typedef float real;            // 使用单精度浮点数
    const real EPSILON = 1.0e-6;   // 单精度下的极小值常量
#endif

const real x0 = 100.0;  // 定义目标值常量，用于比较平方根

// CPU版本的算术运算函数
// 参数说明：
// - x: 输入输出数组指针
// - x0: 目标比较值
// - N: 数组元素个数
void arithmetic(real *x, const real x0, const int N)
{
    // 遍历数组中的每个元素
    for(int n = 0; n < N; n++)
    {
        real x_tmp = x[n];  // 获取当前元素值
        
        // 循环增加x_tmp直到其平方根大于等于x0
        while(sqrt(x_tmp) < x0)
        {
            ++x_tmp;  // 逐步增加数值
        }
        
        x[n] = x_tmp;  // 将计算结果存回原数组位置
    } 
}

// GPU版本的CUDA内核函数
// 参数说明：
// - d_x: 设备端数组指针
// - x0: 目标比较值
// - N: 数组元素总数
void __global__ arithmetic(real *d_x, const real x0, const int N)
{
    // 计算当前线程的全局索引
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 边界检查：确保线程索引不超出数组范围
    if(n < N)
    {
        real x_tmp = d_x[n];  // 获取当前线程负责的数组元素
        
        // 循环增加x_tmp直到其平方根大于等于x0
        while(sqrt(x_tmp) < x0)
        {
            ++x_tmp;  // 逐步增加数值
        }
        
        d_x[n] = x_tmp;  // 将计算结果写回设备数组
    }
}