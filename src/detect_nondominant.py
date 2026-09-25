# -*- coding: utf-8 -*-
"""
detect_nondominant.py —— 非惯用手（PPG）分支【接口定义】
========================================================
负责人：鲁焯俊 ｜ 接口约定：黄嘉俊 ｜ 更新：2026-09-25

【作用】
    手表戴在非惯用手时，手部动作弱，改用 PPG（光电容积脉搏波）生理信号检测进食。
    与惯用手分支（六轴 IMU）是两条独立路线，但**共用**底层数据读取与评测。

【两个分支的统一约定】（对接用，请遵守）
    输入: 某受试者的传感器数据（DataFrame，从 data_loader 读出来）
    输出: [(start_ms, end_ms), ...]  —— 检测到的进食事件段，毫秒时间戳
    评测: 统一调用 evaluate.py（事件级 IoU > 0.25 → P/R/F1）

【⚠️ 关键区分：训练期 vs 测试期】
    - 训练/调参期: 可以拿到真值饭段（用来定阈值、训练模型）
    - 测试期:      ⚠️ 绝对不能看真值 —— 看了就是数据泄漏，分数会被判无效
    所以下面接口里的 meals 参数是可选的：测试时传 None。

【可以复用的部分】（不用重写）
    - `data_loader.load_sensor_zip()` 读 zip（PPG 和 IMU 在同一个文件里）
    - `subjects.py`（受试者名册 / ID 规范化 / 惯用手标记）
    - `preprocess.py` 里的 `make_labels()` 标签逻辑（与信号类型无关）
    - `evaluate.py` 评测（IoU + P/R/F1）
    - `output.py` 统一输出

【需要你实现的两个函数】
"""


def extract_ppg_features(df, window_ms=30000, step_ms=15000):
    """
    从 PPG 信号中提取特征（这一层你自由设计）。

    参数
    ----
    df : pd.DataFrame
        该受试者的传感器数据，含 PPG_* 列与时间戳列
    window_ms : int
        窗口长度（毫秒）。PPG 建议 30 秒窗（你也可按实验调整）
    step_ms : int
        滑窗步长（毫秒）

    返回
    ----
    X : np.ndarray, shape (n_windows, n_features)
        每个窗口的特征向量
    starts : np.ndarray, shape (n_windows,)
        每个窗口的起点时间戳（毫秒）—— 后续切段、打标签都要用它

    提示
    ----
    · 特征可以做：信号质量类、心率类、形态类、时序上下文类（参考你的 PPT 口径）
    · 44 通道可以做筛选/汇总（比如取质量最好的若干个）
    · ⚠️ 记得处理采集空档（参考 preprocess 里的断点切段思路）
    """
    # TODO（鲁焯俊）


def detect_nondominant(df, meals=None):
    """
    非惯用手分支的主入口 —— 整合时会被统一入口调用。

    参数
    ----
    df : pd.DataFrame
        该受试者的完整传感器数据
    meals : list of (int, int) or None
        真值饭段列表。**训练/调参期可传**（用于定阈值/训练）；
        **测试期必须传 None**（不能看真值）。

    返回
    ----
    segments : list of (int, int)
        检测到的进食事件段 [(start_ms, end_ms), ...]
        空列表表示没检测到 —— 不要返回 None。

    实现建议（按你的规则方法）
    --------------------------
    1) X, starts = extract_ppg_features(df)
    2) 按你的多条件规则（或模型）在每个窗口上判断
    3) 把连续的判定窗口合并成事件段（参考 segment_meals.py 的做法）
    4) 后处理：平滑 → 阈值 → 间隔合并
    5) return [(start_ms, end_ms), ...]
    """
    # TODO（鲁焯俊）


if __name__ == "__main__":
    # ===== 你的自测代码放这里 =====
    # 建议至少验证：
    #   ① 单受试者能跑通，输出是 [(int, int), ...] 格式
    #   ② 用 evaluate.py 算一次 F1（先自己人验证，正式评估要按受试者分组）
    print("detect_nondominant 尚未实现 —— 这里是自测入口")
