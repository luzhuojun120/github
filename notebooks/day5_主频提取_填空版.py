"""
Day 5 挑战：主频提取（FFT 找最明显的节奏）
目标：输入一个窗口信号 → 输出它的主频（Hz）
前置：刚看过演示——FFT 把信号拆成"各频率分量"，幅度最大的频率 = 主频
运行：直接 python 跑，看到 主频 ≈ 3.0 Hz 即成功
"""

import numpy as np


def dominant_freq(window, fs):
    """
    提取窗口的主频（幅度最大的频率）
    参数:
        window: 一维数组（一个窗口的信号值）
        fs: 采样率（每秒多少个点，决定频率刻度）
    返回:
        主频（Hz，float）
    """
    n = len(window)

    # ===== TODO 1: 做 FFT =====
    # 提示：np.fft.fft(window)  → 返回复数数组（每个频率一个复数值）
    fft_vals = np.fft.fft(window)                      # ← 填 FFT

    # ===== TODO 2: 取幅度（复数 → 实数大小）=====
    # 提示：np.abs(fft_vals) → 每个频率的分量大小（复数取模）
    amps = np.abs(fft_vals)# ← 填 abs

    # ===== TODO 3: 频率刻度 + 找幅度最大的频率 =====
    # 提示1：np.fft.fftfreq(n, 1/fs) 给每个位置对应的 Hz
    # 提示2：amps[0] 是 0Hz（直流/平均水平，不算节奏），从 1 开始找
    # 提示3：np.argmax(数组) 返回最大值的下标
    freqs = np.fft.fftfreq(n, 1 / fs)
    peak_idx = np.argmax(amps[1:]) + 1   # +1 是因为跳过了 0Hz
    return abs(freqs[peak_idx])          # 频率刻度上主峰的位置 = 主频


# ===== 测试：造一个已知节奏的信号（3Hz，采样率 100Hz，2 秒）=====
fs = 100
t = np.arange(200) / fs
window = np.sin(2 * np.pi * 3 * t) + 0.3 * np.random.randn(200)   # 3Hz 信号 + 噪声

peak = dominant_freq(window, fs)
print(f"提取到的主频: {peak:.2f} Hz")
print(f"期望值:       ≈ 3.00 Hz（信号是 3Hz 造的，噪声不该骗过你）")
