"""
Day 4 开胃挑战：窗口特征提取（把每个窗口"压缩"成一行特征）
目标：对每个窗口算 [均值, 标准差, 最大值, 最小值] → 输出 (窗口数, 4)
为什么需要这步：30 个原始值太"原始"，统计量才有区分度
              （UCI HAR 的 561 列 = 对每个窗口提取一堆这样的统计量）
运行：直接 python 跑，看到 shape=(8, 4) 且 feat[0] ≈ [14.5, 8.8, 29, 0] 即成功
"""

import numpy as np

# 直接复用你 Day 3 写好的滑窗函数（抄过来用，或者 import 都行）
def sliding_windows(signal, w, s):
    windows = []
    start = 0
    while start + w <= len(signal):
        windows.append(signal[start:start + w])
        start += s
    return np.array(windows)


def window_stats(windows):
    """
    对每个窗口算 4 个统计特征
    参数:
        windows: 二维数组 shape=(窗口数, 窗长)（Day 3 的产物）
    返回:
        二维数组 shape=(窗口数, 4)，每行 = [mean, std, max, min]
    """
    stats = []
    for w in windows:                # w = 一个窗口（30 个数）
        # ===== TODO 1: 算这个窗口的均值 =====
        mean = np.mean(w)                   # 提示: np.mean(w)
        # ===== TODO 2: 算标准差（波动程度）=====
        std = np.std(w)                    # 提示: np.std(w)
        # ===== TODO 3: 算最大值和最小值 =====
        mx = np.max(w)                     # np.max(w)
        mn = np.min(w)                     # np.min(w)
        stats.append([mean, std, mx, mn])
    return np.array(stats)


# ===== 测试：把 Day 3 的 100 点信号切窗再算特征 =====
ws = sliding_windows(np.arange(100), 30, 10)   # 8 个窗口
feat = window_stats(ws)

print("特征数组形状:", feat.shape)              # 期望 (8, 4)
print("第 1 个窗口特征:", feat[0].round(2))     # 期望 [14.5 8.66 29 0]
print("第 2 个窗口特征:", feat[1].round(2))     # 期望 [24.5 8.66 39 10]

# ===== 想一想 =====
# 窗口1 是 0~29，窗口2 是 10~39：为什么它们的 std 一样（8.66）？
# 提示：标准差看"离散程度"，跟数据整体平移无关
