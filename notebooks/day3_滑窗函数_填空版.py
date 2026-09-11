"""
Day 3 练习：手写滑动窗口函数（把一整条信号切成样本）
目标：输入一维信号 → 输出 (窗口数 × 窗长) 的二维数组
前置：刚学过公式 窗口数 = (N - w) // s + 1
运行：直接 python 跑，看到 shape=(8, 30) 即成功
"""

import numpy as np


def sliding_windows(signal, w, s):
    """
    把一维信号切成若干窗口
    参数:
        signal: 一维数组（一整条连续信号）
        w: 窗口长度（每个窗口看几个点）
        s: 步长（相邻窗口起点往后挪几个点）
    返回:
        二维数组，shape = (窗口数, w)，每行是一个窗口
    """
    N = len(signal)
    windows = []

    start = 0
    # ===== TODO 1: 循环条件 =====
    # 提示：窗口起点 start 要满足"窗口能放下"：start + w 不能超过 N
    # 仿照刚才演示里的 while 条件
    while start+w<=N:                     # ← 填条件
        # ===== TODO 2: 切出窗口 =====
        # 提示：取 signal 从 start 到 start+w 这一段（切片 [a:b] 含 a 不含 b）
        window = signal[start:start+w]             # ← 填切片
        windows.append(window)

        # ===== TODO 3: 起点往后挪 =====
        start=start+s                       # ← 填 start = start + s

    return np.array(windows)


# ===== 测试：N=100 的假信号，信号值就是 0,1,2,...,99（方便对答案）=====
signal = np.arange(100)

ws = sliding_windows(signal, 30, 10)

# ===== 验收标准（4 条全中 = 过关）=====
print("窗口数组形状:", ws.shape)             # 期望 (8, 30)
print("第 1 个窗口前 5 个值:", ws[0][:5])     # 期望 [0 1 2 3 4]
print("第 2 个窗口前 3 个值:", ws[1][:3])     # 期望 [10 11 12]  ← 起点挪了 10
print("最后一个窗口后 3 个值:", ws[-1][-3:])   # 期望 [97 98 99]  ← 结尾没丢数据
