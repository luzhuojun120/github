"""
features.py
===========
特征工程模块（参赛正式代码 v0.1 · 2026-09-08）

作用：
    把滑窗切出的小段（每个窗口 = 一小段信号）压缩成"特征向量"，
    供分类模型判断这个窗口是"在吃饭"还是"没吃饭"。

设计原则：
    - 每个窗口 → 一行特征（mean/std/max/min = 时域；主频 = 频域）
    - 窗口越大越像"形状描述"，越能区分动作 vs 静止
    - v0.1 先做 5 个特征，跑通后可以无限加（工程上叫"特征扩展"）

模块内函数：
    window_features(window, fs)
        单个窗口 → 特征向量 [mean, std, max, min, 主频]
    extract_features(windows, fs)
        所有窗口 → 特征矩阵 (n_windows, 5)

用法示例：
    from data_loader import load_sensor_zip
    from preprocess import sliding_window
    from features import extract_features

    acc = load_sensor_zip("xxx.zip")          # 读 ACC/GYRO
    x   = acc['ACC_X'].to_numpy()            # 取一条通道
    ws  = sliding_window(x, 500, 250)         # 切窗 (窗口数, 500)
    F   = extract_features(ws, fs=105)        # 特征矩阵 (窗口数, 5)

作者：品食Panda 队 · 假菌
"""

import numpy as np

# ACC 实际采样率 ~105Hz（格式分析结论），频域特征需要它
ACC_FS = 105.0


def window_features(window, fs=ACC_FS):
    """
    单个窗口 → 特征向量。

    参数
    ----
    window : np.ndarray, shape (window_len,)
        一个窗口的信号值（如某通道 500 个采样点）
    fs : float
        采样率（Hz），算主频用

    返回
    ----
    np.ndarray, shape (5,) → [mean, std, max, min, 主频Hz]
    """
    mean = np.mean(window)
    std = np.std(window)
    mx = np.max(window)
    mn = np.min(window)

    n = len(window)
    fft_vals = np.fft.fft(window)
    amps = np.abs(fft_vals)
    freqs = np.fft.fftfreq(n, 1 / fs)
    peak_idx = np.argmax(amps[1:]) + 1
    dom=abs(freqs[peak_idx])
    return np.array([mean,std,mn,mx,dom])



def extract_features(windows, fs=ACC_FS):
    """
    所有窗口 → 特征矩阵。

    参数
    ----
    windows : np.ndarray, shape (n_windows, window_len)
        滑窗结果（来自 preprocess.sliding_window）
    fs : float
        采样率（Hz）

    返回
    ----
    np.ndarray, shape (n_windows, 5)
        每行 = 一个窗口的特征
    """
    stats=[]
    for w in windows:
        stats.append(window_features(w,fs))
    return np.array(stats)

if __name__ == "__main__":
    # ===== 自测 =====
    # 造两种窗口：一种波动大(像在动)、一种平稳(像静止)
    rng = np.random.default_rng(42)
    moving = rng.normal(0, 1, 500) * 10      # 大幅波动
    still = rng.normal(0, 1, 500) * 0.1      # 几乎静止

    f1 = window_features(moving)
    f2 = window_features(still)
    print("波动窗口特征:", f1.round(2))
    print("   → std 应该很大(>5)，主频无明显意义")
    print("静止窗口特征:", f2.round(2))
    print("   → std 应该很小(<0.5)")
