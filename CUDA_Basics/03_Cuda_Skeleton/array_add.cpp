#include <math.h>   // 包含数学函数库
#include <stdio.h>  // 包含标准输入输出库
#include <stdlib.h> // 包含标准库函数，如malloc和free

// 定义一个很小的浮点数，用于比较两个浮点数是否相等
const double EPSILON = 1.0e-15;
// 定义常量a和b，作为数组x和y的初始值
const double a = 1.23;
const double b = 2.34;
// 定义常量c，作为期望的结果值（a + b）
const double c = 3.57;

// 声明add函数，用于将两个数组x和y相加，结果存储在数组z中
void add(const double *x, const double *y, double *z, int N);
// 声明check函数，用于检查数组z中的值是否等于期望值c
void check(const double *z, int N);

int main(void)
{
    // 定义数组大小N
    const int N = 100000000;
    // 计算数组所需的内存大小M（以字节为单位）
    const int M = sizeof(double) * N;
    
    // 动态分配内存给数组x、y和z
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);

    // 初始化数组x和y，所有元素都设置为常量a和b
    for (int n = 0; n < N; n++)
    {
        x[n] = a;
        y[n] = b;
    }

    // 调用add函数，将数组x和y相加，结果存储在数组z中
    add(x, y, z, N);
    // 调用check函数，验证数组z中的值是否正确
    check(z, N);

    // 释放动态分配的内存
    free(x);
    free(y);
    free(z);

    return 0;
}

// 实现add函数：将数组x和y逐元素相加，结果存储在数组z中
void add(const double *x, const double *y, double *z, int N)
{
    for (int n = 0; n < N; n++)
    {
        z[n] = x[n] + y[n];
    }
}

// 实现check函数：检查数组z中的每个元素是否等于期望值c
void check(const double *z, int N)
{
    bool has_errors = false;  // 标志位，表示是否有错误
    for (int n = 0; n < N; n++)
    {
        // 如果z[n]与c的差值大于EPSILON，则认为存在误差
        if (fabs(z[n] - c) > EPSILON)
        {
            has_errors = true;
        }
    }
    // 输出检查结果：如果有错误则输出"Has errors"，否则输出"No errors"
    printf("%s\n", has_errors ? "Has errors" : "No errors");
}