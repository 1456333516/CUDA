#pragma once // 防止头文件被重复包含
#include <stdio.h> // 包含标准输入输出库，用于打印错误信息

/* 
 * 使用方法：CHECK(cudaFree(ptr));
 * 使用上述方法不能捕抓调用核函数相关错误，因为核函数不返回任何值，因为是用void修饰
 * 可以用以下方法
 * 即在调用核函数之后加上如下两条语句
 * CHECK(cudaGetLastError());
 * CHECK(cudaDeviceSynchronize());
 * 但是会导致性能下降，一般调试用
 */
// 
// 在定义宏时，如果一行写不下，则需要在行末写\，表示续行
// 定义一个宏 CHECK，用于检查 CUDA API 调用是否成功
#define CHECK(call) \
do \
{ \
    /* 执行传入的 CUDA API 调用，并将返回的错误码保存到 error_code 变量中 */ \
    const cudaError_t error_code = call; \
    \
    /* 如果错误码不等于 cudaSuccess，表示调用失败 */ \
    if(error_code != cudaSuccess) \
    { \
        /* 打印详细的错误信息 */ \
        printf("CUDA Error:\n"); \
        printf("    File:       %s\n",__FILE__);         /* 打印发生错误的源文件名 */ \
        printf("    Line:       %d\n",__LINE__);         /* 打印发生错误的行号 */ \
        printf("    Error code: %d\n",error_code);      /* 打印错误码 */ \
        printf("    Error text: %s\n",cudaGetErrorString(error_code)); /* 打印错误描述文本 */ \
        \
        /* 终止程序执行 */ \
        exit(1); \
    } \
} while (0); // 使用 do-while(0) 结构确保宏的安全性，避免在条件语句中出现语法问题