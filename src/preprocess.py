"""
preprocess.py
=============
数据预处理模块（参赛正式代码 v0.1）

作用：
    把一整条 PPG/ACC 时间序列切成固定长度的小段（滑动窗口），
    再对窗口做归一化，让模型能"一段一段"地学习进食动作。

模块内函数：
    sliding_window(signal, window_size, step)
        把一维信号切成二维数组 (窗口数, 窗口长度)
    normalize(windows, method="zscore")
        对窗口做归一化，消除量纲差异
    make_labels(...)
        （TODO）按饭段时间给每个窗口打"吃/没吃"标签

用法示例：
    from preprocess import sliding_window, normalize
    windows = sliding_window(signal, window_size=500, step=250)
    windows = normalize(windows)

作者：品食Panda 队 · 假菌
"""

import numpy as np


def sliding_window(signal, window_size, step):
    """
    把一维信号切成若干等长窗口。

    参数
    ----
    signal : np.ndarray, shape (N,)
        一整条连续信号（如某条 ACC 通道的所有采样点）
    window_size : int
        每个窗口包含几个采样点（窗口长度）
    step : int
        相邻窗口起点之间相隔几个采样点（步长）

    返回
    ----
    np.ndarray, shape (n_windows, window_size)
        每行是一个窗口。n_windows = (N - window_size) // step + 1

    说明
    ----
    - 窗口起点从 0 开始，每次前进 step，直到放不下一个完整窗口为止
    - 若 signal 长度不足以容纳任何窗口，抛 ValueError
    """
    # >>> TODO(你来实现) <<<
    # 把你在 Day 3 写的逻辑放进来，并加一个"太短就报错"的检查：
    # if len(signal) < window_size:
    #     raise ValueError(f"信号长度 {len(signal)} 小于窗口长度 {window_size}")
    if len(signal) < window_size:
        raise ValueError(f"信号长度{len(signal)}小于窗口长度{window_size}")
    N=len(signal)
    windows=[]
    start=0
    while start+window_size<=N:
        window=signal[start:window_size+start]
        windows.append(window)
        start=start+step
    return np.array(windows)



def normalize(windows, method="zscore"):
    """
    对每个窗口做归一化。

    参数
    ----
    windows : np.ndarray, shape (n_windows, window_size)
        滑窗后的二维数组（来自 sliding_window）
    method : str
        "zscore": 每窗口减去均值、除以标准差（默认）
        "minmax": 每窗口缩放到 [0, 1]

    返回
    ----
    np.ndarray，形状与输入相同
    """
    if method == "zscore":
        # axis=1 表示"沿每个窗口（行）内部算"
        mu = windows.mean(axis=1, keepdims=True)
        sd = windows.std(axis=1, keepdims=True)
        # 防止某个窗口完全静止（std=0）导致除零
        sd = np.where(sd < 1e-9, 1.0, sd)
        return (windows - mu) / sd
    elif method == "minmax":
        lo = windows.min(axis=1, keepdims=True)
        hi = windows.max(axis=1, keepdims=True)
        rng = np.where(hi - lo < 1e-9, 1.0, hi - lo)
        return (windows - lo) / rng
    else:
        raise ValueError(f"未知的 method: {method}（可选 'zscore' / 'minmax'）")


def make_labels(window_starts,window_len, eat_events):
    """
    （TODO · 下次会话实现）
    给每个窗口打"吃 / 没吃"标签。

    参数（预计）
    ----
    window_starts : 每个窗口起点对应的时间戳
    eat_events : 饭段时间列表，如 [(start_ts1, end_ts1), (start_ts2, end_ts2), ...]

    返回
    ----
    np.ndarray, shape (n_windows,)，1 = 该窗口属于某顿饭，0 = 非进食
    """
    labels = []
    for start in window_starts:
        a = start
        b = start + window_len
        is_eat = False
        for (c, d) in eat_events:
            if b > c and d > a:
                is_eat = True
                break
        labels.append(1 if is_eat else 0)
        # if is_eat == True:
        #    labels.append(1)
        # else:
        #    labels.append(0)

    return np.array(labels)




if __name__ == "__main__":
    # ===== 自测：跑通即模块可用 =====
    signal = np.arange(100.0)          # 假信号 0~99
    ws = sliding_window(signal, 30, 10)
    print("sliding_window 输出形状:", ws.shape)          # 期望 (8, 30)

    z = normalize(ws)
    print("归一化后第 1 窗口:", z[0].round(3))            # 期望 mean≈0, std≈1
    print("  校验 mean≈0:", round(float(z[0].mean()), 6),
          " std≈1:", round(float(z[0].std()), 3))
    eat = [(650,750)]
    start=[400,500,600,700,800]
    labels=make_labels(start,window_len=100,eat_events=eat)
    print("标签:",labels)