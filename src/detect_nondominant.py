"""
detect_nondominant.py
=====================
非惯用手场景检测（队友主责调研，你辅助）。

作用：
    手表戴在非惯用手时，手部动作消失，
    改用 PPG（光电心率）生理信号（进食时心率变化）来判断进食。

将来要写的函数：
    - extract_ppg_features(window): 从 PPG 提取特征（如心率变异性 HRV）
    - detect_nondominant(window): 返回该窗口是否进食（True / False）

注意：
    这条路线和惯用手完全不同，不能复用 IMU 的代码。
"""

# TODO: 实现 extract_ppg_features
# TODO: 实现 detect_nondominant
