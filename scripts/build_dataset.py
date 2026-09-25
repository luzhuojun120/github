"""build_dataset.py---把真实的senor.zip变成训练数据(x,y)"""
import sys,os
WINDOW=500
STEP=250




_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
import numpy as np
import pandas as pd
from data_loader import load_sensor_zip
from preprocess import sliding_window, make_labels
from features import extract_features, ACC_FS
from paths import RAW
def build_dataset(subject="HNU21026"):
    map_df = pd.read_csv(os.path.join(RAW, "sensor_下载映射表.csv"), encoding='utf-8-sig')
    print(map_df.head(), map_df.shape)
    hnu = map_df[map_df['externalid'] == "HNU21026"]
    print("HNU21026的zip数：", len(hnu))
    map_df1 = pd.read_csv(os.path.join(RAW, "mealinfo_标注表.csv"))
    print(map_df1.head(), map_df1.shape)
    hnu1 = map_df1[map_df1['externalid'] == "HNU21026"]
    print("HNU21016的zip数：", len(hnu1))
    zip_starts = hnu['timeStamp.startTime'].tolist()
    zip_ends = hnu['timeStamp.endTime'].tolist()
    print("zip起：", zip_starts[:3])
    meal_starts = hnu1['beforeTime'].tolist()
    meal_ends = hnu1['afterTime'].tolist()
    print("meal起：", meal_starts[:3])
    if meal_starts[0] > zip_starts[0] and meal_ends[0] < zip_ends[0]:
        print("第一餐在第零个zip里")
    else:
        print("第一餐不在第零个zip里")
    best_idx = 0
    best_cnt = -1
    for i in range(len(zip_starts)):
        cnt = 0
        for j in range(len(meal_starts)):
            if meal_starts[j] > zip_starts[i] and meal_ends[j] < zip_ends[i]:
                cnt += 1
        print(f"zip{i}覆盖meal{cnt}")
        if cnt > best_cnt:
            best_cnt = cnt
            best_idx = i
    zip_path = hnu['sensorData'].tolist()
    chosen = zip_path[best_idx]
    print("选中zip", chosen)
    print("选zip", best_idx, "覆盖", best_cnt, "餐")
    local_zip = os.path.join(RAW, "sensorData", os.path.basename(chosen))
    acc = load_sensor_zip(local_zip)
    print("读入采样：", acc.shape)
    x = acc['ACC_X'].to_numpy()
    t = acc['ACC_TIME'].to_numpy()
    print("信号点数：", len(x), "|时间点数：", len(t))
    windows = sliding_window(x, WINDOW, STEP)
    print("窗口数：", windows.shape)
    starts = sliding_window(t, WINDOW, STEP)[:, 0]
    print("窗口开始时刻前三个:", starts[:3])
    X = extract_features(windows)
    print("特征矩阵：", X.shape)
    win_ms = (WINDOW / ACC_FS) * 1000
    print("一个窗口=", win_ms, "毫秒")
    meals = list(zip(meal_starts, meal_ends))
    print("餐列表前两个：", meals[:2])
    y = make_labels(starts, win_ms, meals)
    print("标签发布：吃饭", int(y.sum()), "窗/非吃饭", int((y == 0).sum()), "窗")
    return X,y

if __name__ == "__main__":
    X, y = build_dataset()
    print("X:", X.shape, "y分布：", int(y.sum()), "/", len(y))
