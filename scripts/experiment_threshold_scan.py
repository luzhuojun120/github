"""experiment_threshold_scan.py —— 深度 × 阈值 联合扫描

【目的】
    上一轮实验发现：限深（max_depth 变小）会让概率变「扁平」，
    在固定阈值 0.4 下虚报爆炸 → F1 腰斩。

    本轮验证两个假设：
      H1：完整模型的最优阈值可能不是 0.4（我们一直沿用，没验证过）
      H2：限深模型若同时调高阈值，可能换回性能 → 小模型也能用

【铁律】
    验证集 = 从「训练集内部」切出，绝不用已污染的 7 人测试集。

【运行】
    python scripts/experiment_threshold_scan.py
"""
import os
import sys
import gzip
import time
import random
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from paths import CACHE
from train_eval import load_all_cache, eval_event, match_count

VAL_SIZE = 5
SEED = 0
THRESHS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

CONFIGS = [
    ("完整(深度不限)", dict(n_estimators=100)),
    ("max_depth=25",   dict(n_estimators=100, max_depth=25)),
    ("max_depth=20",   dict(n_estimators=100, max_depth=20)),
    ("max_depth=15",   dict(n_estimators=100, max_depth=15)),
]


def gz_size(clf):
    return len(gzip.compress(pickle.dumps(clf, protocol=4), 6)) / 1e6


def eval_at(clf, data, val_ids, thresh):
    tp = fp = fn = 0
    for sid in val_ids:
        pred_segs, true_segs = eval_event(clf, data[sid], thresh=thresh)
        mc = match_count(pred_segs, true_segs)
        tp += mc
        fp += len(pred_segs) - mc
        fn += len(true_segs) - mc
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def main():
    data = load_all_cache(CACHE)
    ids = sorted(data.keys())
    random.Random(SEED).shuffle(ids)
    val_ids = ids[:VAL_SIZE]
    tr_ids = ids[VAL_SIZE:]
    print(f"训练 {len(tr_ids)} 人 / 验证 {VAL_SIZE} 人 {sorted(val_ids)}")

    X_tr = np.vstack([data[i]["X"] for i in tr_ids])
    y_tr = np.concatenate([data[i]["y"] for i in tr_ids])
    print(f"训练矩阵: {X_tr.shape}\n")

    print(f"{'配置':<18}{'gzip MB':>8} " + "".join(f"{('t='+str(t)):>9}" for t in THRESHS))
    print("-" * 82)

    summary = []
    for name, kw in CONFIGS:
        t0 = time.time()
        clf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **kw)
        clf.fit(X_tr, y_tr)
        mb = gz_size(clf)
        f1s = []
        for th in THRESHS:
            P, R, F = eval_at(clf, data, val_ids, th)
            f1s.append(F)
        best_i = int(np.argmax(f1s))
        summary.append((name, mb, f1s, THRESHS[best_i], f1s[best_i]))
        print(f"{name:<18}{mb:>8.1f} " + "".join(f"{f:>9.4f}" for f in f1s)
              + f"   [最优 t={THRESHS[best_i]} F1={f1s[best_i]:.4f}]  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 82)
    print(f"{'配置':<18}{'gzip MB':>9}{'最优阈值':>10}{'最优F1':>10}{'原始F1(t=0.4)':>15}{'提升':>9}")
    print("-" * 82)
    base = summary[0]
    for name, mb, f1s, bt, bf in summary:
        f04 = f1s[THRESHS.index(0.4)]
        print(f"{name:<18}{mb:>9.1f}{bt:>10}{bf:>10.4f}{f04:>15.4f}{bf-f04:>+9.4f}")


if __name__ == "__main__":
    main()
