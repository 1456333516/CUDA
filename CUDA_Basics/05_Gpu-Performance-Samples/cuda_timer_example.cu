// 声明两个 CUDA 事件对象，分别用于标记计时的起始点和结束点
cudaEvent_t start, stop;

// 创建 CUDA 事件对象 start，用于记录计时开始的时间点
CHECK(cudaEventCreate(&start));

// 创建 CUDA 事件对象 stop，用于记录计时结束的时间点
CHECK(cudaEventCreate(&stop));

// 记录当前时间到事件 start，标志着计时正式开始
CHECK(cudaEventRecord(start));

// 查询事件 start 是否已经完成（非阻塞操作，通常用于检查事件状态）
cudaEventQuery(start);

// ==================== 需要计时的代码块 ====================
// 在此处插入需要测量执行时间的 CUDA 代码
// 例如：内核函数调用、内存拷贝或其他 GPU 操作
// ==========================================================

// 记录当前时间到事件 stop，标志着计时结束
CHECK(cudaEventRecord(stop));

// 等待事件 stop 完成，确保所有之前的 GPU 操作均已执行完毕（阻塞操作）
CHECK(cudaEventSynchronize(stop));

// 声明一个浮点变量，用于存储从 start 到 stop 的经过时间（单位：毫秒）
float elapsed_time;

// 计算从事件 start 到事件 stop 之间的时间差，并将结果存储到 elapsed_time 中
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

// 打印测量得到的时间，格式为 "Time = X ms."
printf("Time = %g ms.\n", elapsed_time);

// 销毁事件对象 start，释放相关资源
CHECK(cudaEventDestroy(start));

// 销毁事件对象 stop，释放相关资源
CHECK(cudaEventDestroy(stop));