"""
run_pipeline_demo.py
===================
端到端数据管线演示：真实 sensor zip → 训练数据 (X 特征, y 标签)
（把已写的 3 个模块串起来 —— preprocess / data_loader / features）

流程：
    1. 挑一个受试者(HNU21026)的 zip，覆盖其某顿饭的时间段
    2. load_sensor_zip 读 ACC_X 通道（~105Hz 时间序列）
    3. sliding_window 切成窗口（500点 ≈ 4.8秒）
    4. extract_features 每窗口 → 5 特征
    5. make_labels 对照饭段时间戳 → 0/1 标签
    6. 输出：能喂模型的数据 + 吃饭 vs 平时特征对比

运行：python scripts/run_pipeline_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from data_loader import load_sensor_zip
from preprocess import sliding_window, make_labels
from features import extract_features, ACC_FS

DATA = r"E:\workbuddy\进食检测比赛\data\raw"
MAP_CSV = os.path.join(DATA, "sensor_下载映射表.csv")
MEAL_CSV = os.path.join(DATA, "mealinfo_标注表.csv")

# ===== 参数 =====
SUBJECT = "HNU21026"      # 有数据 + 有标注的受试者
WINDOW = 500              # 窗口 500 点 ≈ 4.8 秒（@105Hz）
STEP = 250                # 步长（50% 重叠）


def pick_zip_for_meal(subject_rows, meals):
    """选一个覆盖最多饭段时间的 zip"""
    best, best_cnt = None, -1
    for r in subject_rows:
        zip_start, zip_end = int(r[4]), int(r[5])   # 映射表: 起止毫秒时间戳
        cnt = sum(1 for (bs, be) in meals if bs >= zip_start and be <= zip_end)
        if cnt > best_cnt:
            best, best_cnt = os.path.join(DATA, "sensorData", os.path.basename(r[1])), cnt
    return best


def main():
    # ---- 1. 找受试者数据 ----
    map_rows = [r for r in pd.read_csv(MAP_CSV, encoding="utf-8-sig").values if r[0] == SUBJECT]
    meals = [(int(r[2]), int(r[3])) for r in
             pd.read_csv(MEAL_CSV, encoding="utf-8-sig").values if r[0] == SUBJECT]
    print(f"{SUBJECT}: {len(map_rows)} 个 zip, {len(meals)} 餐")
    print(f"餐段时间样例: {meals[:2]}")

    # ---- 2. 读 ACC_X ----
    zip_path = pick_zip_for_meal(map_rows, meals)
    print(f"\n选用 zip: {os.path.basename(zip_path)}")
    acc = load_sensor_zip(zip_path)
    t = acc["ACC_TIME"].to_numpy()
    x = acc["ACC_X"].to_numpy()
    print(f"ACC_X 采样: {len(x)} 点, 时间跨度 {(t[-1]-t[0])/1000:.0f} 秒")

    # ---- 3. 切窗 ----
    windows = sliding_window(x, WINDOW, STEP)
    # 每个窗口的"开始时刻" = 该窗口第 1 行的 ACC_TIME
    starts = sliding_window(t, WINDOW, STEP)[:, 0]
    print(f"窗口数: {len(windows)}, 每窗 {WINDOW} 点")

    # ---- 4. 特征 ----
    X = extract_features(windows)
    print(f"特征矩阵: {X.shape} (窗口数 × 5特征)")

    # ---- 5. 标签（窗口长换算成毫秒，与饭段时间戳同单位）----
    win_ms = (WINDOW / ACC_FS) * 1000     # 500点/105Hz ≈ 4762ms
    y = make_labels(starts, win_ms, meals)
    print(f"标签分布: 吃饭 {y.sum()} 窗 / 非吃饭 {(y==0).sum()} 窗")

    # ---- 6. 吃饭 vs 平时 特征对比（检验特征区分力）----
    print("\n===== 特征均值对比 (吃饭窗口 vs 非吃饭窗口) =====")
    feat_names = ["mean", "std", "max", "min", "主频Hz"]
    for i, name in enumerate(feat_names):
        eat = X[y == 1][:, i].mean()
        not_eat = X[y == 0][:, i].mean()
        mark = " ← 有区分!" if abs(eat - not_eat) > 0.3 * max(abs(eat), abs(not_eat), 1) else ""
        print(f"{name:6s}: 吃饭 {eat:9.2f} | 平时 {not_eat:9.2f}{mark}")


if __name__ == "__main__":
    main()
