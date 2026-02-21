// 版本一：有返回值的设备函数
// 定义一个设备函数 add1_device，接收两个 double 类型参数 x 和 y，
// 返回它们的和。此函数在 GPU 设备上执行。
double __device__ add1_device(const double x, const double y)
{
    return x + y; // 直接返回 x 与 y 的加法结果
}

// 全局函数 add1，用于启动 GPU 上的并行计算。
// 参数说明：
// - x: 输入数组 x 的指针
// - y: 输入数组 y 的指针
// - z: 输出数组 z 的指针，存储 x + y 的结果
// - N: 数组长度
void __global__ add1(const double *x, const double *y, double *z, const int N)
{
    // 计算当前线程的全局索引
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 确保线程索引不超出数组边界
    if (n < N)
    {
        // 调用设备函数 add1_device 计算 x[n] + y[n]，并将结果存入 z[n]
        z[n] = add1_device(x[n], y[n]);
    }
}

// 版本二：用指针的设备函数
// 定义一个设备函数 add2_device，接收两个 double 类型参数 x 和 y，
// 以及一个指向 double 的指针 z，将 x + y 的结果写入 *z。
void __device__ add2_device(const double x, const double y, double *z)
{
    *z = x + y; // 将 x + y 的结果写入指针 z 指向的内存位置
}

// 全局函数 add2，功能与 add1 类似，但调用的是 add2_device。
void __global__ add2(const double *x, const double *y, double *z, const int N)
{
    // 计算当前线程的全局索引
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 确保线程索引不超出数组边界
    if (n < N)
    {
        // 调用设备函数 add2_device，传入 x[n]、y[n] 和 &z[n] 的地址
        add2_device(x[n], y[n], &z[n]);
    }
}

// 版本三：用引用的设备函数
// 定义一个设备函数 add3_device，接收两个 double 类型参数 x 和 y，
// 以及一个 double 类型的引用 z，将 x + y 的结果赋值给 z。
void __device__ add3_device(const double x, const double y, double &z)
{
    z = x + y; // 将 x + y 的结果赋值给引用 z
}

// 全局函数 add3，功能与前两个版本类似，但调用的是 add3_device。
void __global__ add3(const double *x, const double *y, double *z, const int N)
{
    // 计算当前线程的全局索引
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 确保线程索引不超出数组边界
    if (n < N)
    {
        // 调用设备函数 add3_device，传入 x[n]、y[n] 和 z[n] 的引用
        add3_device(x[n], y[n], z[n]);
    }
}