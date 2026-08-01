"""
preprocess.py
=============
数据预处理模块。

作用：
    把长条的时间序列切成固定长度的小段（滑动窗口），并做归一化，
    让模型能"一段一段"地学习进食动作。

将来要写的函数：
    - sliding_window(signal, window_size, step): 按窗口长度 + 步长切窗
    - normalize(windows): 归一化到 0~1 或 z-score
    - make_labels(windows, eat_events): 给每段打标签（进食 / 非进食）

概念：
    采样率 = 每秒采多少个点；窗口长度通常按"秒 × 采样率"算。
"""

# TODO: 实现 sliding_window
# TODO: 实现 normalize
# TODO: 实现 make_labels
