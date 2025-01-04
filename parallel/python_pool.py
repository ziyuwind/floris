# 特性	multiprocessing.Pool	MPI
# 通信方式	共享内存或本地通信	消息传递，适用于分布式环境
# 数据传递效率	快（共享内存）	慢（需序列化和网络传输）
# 跨节点支持	不支持，仅限单机	支持多节点，适合分布式环境

# multiprocessing.Pool
# 工作模式：基于共享内存模型，主要在单台机器（节点）上并行计算。
# 通信方式：通过共享内存或队列在进程之间传递数据。
# 易用性：Python 原生支持，安装简单，语法容易理解。
# 适用范围：
# 单机多核并行计算。
# 任务较小，数据可以通过共享内存传递的场景。


# multiprocessing.Pool 是 Python 中用于并行处理任务的一个强大工具。它提供了一种简单的方式来创建进程池并管理任务的分配和执行。以下是 multiprocessing.Pool 的主要用法和示例：

# 基本功能
# 1.创建进程池：可以通过 Pool(processes=n) 创建一个包含 n 个工作进程的进程池。如果 processes 参数未指定，默认使用 CPU 的核心数量。

# 2.分发任务：可以使用以下方法将任务分发到进程池中：

# map(func, iterable)：类似于内置的 map()，将 iterable 中的每个元素作为参数传递给函数 func，并返回结果列表。
# apply(func, args, kwds)：在进程池中执行一个函数，阻塞直到返回结果。
# apply_async(func, args, kwds)：非阻塞地执行一个函数，通过回调函数获取结果。
# starmap(func, iterable)：类似于 map()，但可以将多个参数作为元组传递给 func。
# imap() / imap_unordered()：生成器版本的 map()，支持懒加载。
# 3.关闭和等待：

# close()：阻止池接受新的任务。
# join()：等待所有工作进程完成


# 注意事项
# 1.跨平台问题：

# 在 Windows 上，if __name__ == "__main__": 必须包含在主代码中，否则会引发 RuntimeError。
# 2.进程间通信：

# 如果需要共享数据，可以使用 multiprocessing.Manager 提供的共享数据结构，如列表或字典。
# 3.调试：

# 子进程中的错误信息可能不会显示在主进程中，因此调试时可以使用日志或打印信息来捕获异常。
# 通过 Pool，我们可以轻松实现多进程任务的并行处理，从而提高计算效率。

from multiprocessing import Pool

def multiply(a, b):
    return a * b

if __name__ == "__main__":
    with Pool(processes=3) as pool:
        tasks = [(1, 2), (3, 4), (5, 6), (7, 8)]
        results = pool.starmap(multiply, tasks)
        print("Results:", results)