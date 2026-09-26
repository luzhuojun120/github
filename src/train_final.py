"""train_final.py —— 训练最终交付模型 → model.pkl

【和 train_eval.py 的分工】
    train_eval.py   → 留出部分受试者不参与训练，用来评估「方法」能否泛化到新用户
                      （产出的 F1 是报告里的性能数字）
    train_final.py  → 用全部受试者重训，产出真正打包进 exe 的模型
                      （交付时官方测试集是另外的受试者，本地所有人都是可用素材）

【参数单一真相源】
    WINDOW/STEP/ACC_FS/阈值/平滑/合并间隔等参数全部随模型一起存进 model.pkl，
    predict.py 加载后直接读，不在别处重写第二遍 —— 避免「两处口径不一致」。

【运行】
    python src/train_final.py
"""
import os
import gzip
import pickle
import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from paths import CACHE, MODEL
from train_eval import load_all_cache, WINDOW, ACC_FS

# ---- 特征参数（与 dataset.py 保持一致，改一处必须改两处 → 已集中在此） ----
IMU_AXES = ["ACC_X", "ACC_Z", "ACC_Y", "GYRO_X", "GYRO_Y", "GYRO_Z"]
STEP = 250

# ---- 后处理参数（与 train_eval.eval_event 默认值保持一致） ----
THRESH = 0.4              # 平滑后的判正阈值
SIGMA = 3                 # 高斯平滑 sigma
SPLIT_GAP_MS = 300000     # 段内平滑的切分间隔（5 分钟，实测同饭内最大缝隙 2.1 分钟）
MERGE_GAP_MS = 180000     # 饭段合并间隔（180 秒，官方后处理要求）
MIN_POSITIVE = 100        # 正样本少于该数的受试者不参与训练


def train_final(min_positive=MIN_POSITIVE):
    """读全部缓存 → 训练 → 存 model.pkl，返回 payload"""
    data = load_all_cache(CACHE, min_pos=min_positive)
    ids = sorted(data.keys())

    X = np.vstack([data[i]["X"] for i in ids])
    y = np.concatenate([data[i]["y"] for i in ids])

    print(f"参与训练受试者: {len(ids)} 人")
    print(f"训练窗口总数:   {X.shape[0]:,}  | 特征维度: {X.shape[1]}")
    print(f"正样本:         {int(y.sum()):,}  ({y.mean()*100:.2f}%)")

    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    print("训练完成")

    payload = {
        "clf": clf,
        "params": {
            "WINDOW": WINDOW,
            "STEP": STEP,
            "ACC_FS": ACC_FS,
            "IMU_AXES": IMU_AXES,
            "THRESH": THRESH,
            "SIGMA": SIGMA,
            "SPLIT_GAP_MS": SPLIT_GAP_MS,
            "MERGE_GAP_MS": MERGE_GAP_MS,
        },
        "meta": {
            "version": "1.0",
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_subjects": len(ids),
            "subject_ids": ids,
            "n_windows": int(X.shape[0]),
            "n_positive": int(y.sum()),
            "note": "用全部受试者训练（交付版）；报告中的 F1 来自 train_eval.py 的按受试者留出评估",
        },
    }

    # gzip 压缩保存：实测 546 MB → 118 MB（4.6x）。加载时用 gzip.open + pickle.load
    with gzip.open(MODEL, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    size_mb = os.path.getsize(MODEL) / 1024 / 1024
    print(f"已保存: {MODEL}  ({size_mb:.1f} MB)")
    return payload


if __name__ == "__main__":
    train_final()
