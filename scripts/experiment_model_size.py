"""experiment_model_size.py —— 模型瘦身实验：体积 vs 性能

【铁律】
    验证集只能从「参与训练的人」里切出来。
    那 7 个测试人（train_eval.py 的 te）已经看过了 → 再用就是信息泄漏。

【做法】
    37 人 → 随机切 5 人做验证集、32 人做训练集
    对每个复杂度配置：训练 → 在验证集上算事件级 P/R/F1 → 量体积

【运行】
    python scripts/experiment_model_size.py
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


def measure_size(clf):
    """返回 (原始 MB, gzip 后 MB)"""
    raw = pickle.dumps(clf, protocol=4)
    return len(raw) / 1e6, len(gzip.compress(raw, 6)) / 1e6


def eval_on_validation(clf, data, val_ids):
    """事件级微平均（人内匹配，再累加）"""
    tp = fp = fn = 0
    for sid in val_ids:
        pred_segs, true_segs = eval_event(clf, data[sid])
        mc = match_count(pred_segs, true_segs)
        tp += mc
        fp += len(pred_segs) - mc
        fn += len(true_segs) - mc
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F, tp, fp, fn


def main():
    data = load_all_cache(CACHE)
    ids = sorted(data.keys())
    random.Random(SEED).shuffle(ids)
    val_ids = ids[:VAL_SIZE]
    tr_ids = ids[VAL_SIZE:]
    print(f"总受试者 {len(ids)} → 训练 {len(tr_ids)} / 验证 {VAL_SIZE}")
    print(f"验证集: {sorted(val_ids)}\n")

    X_tr = np.vstack([data[i]["X"] for i in tr_ids])
    y_tr = np.concatenate([data[i]["y"] for i in tr_ids])
    print(f"训练矩阵: {X_tr.shape}  正样本 {int(y_tr.sum()):,} ({y_tr.mean()*100:.2f}%)\n")

    configs = [
        ("完整(基准)",        dict(n_estimators=100)),
        ("n_estimators=30",   dict(n_estimators=30)),
        ("max_depth=25",      dict(n_estimators=100, max_depth=25)),
        ("max_depth=15",      dict(n_estimators=100, max_depth=15)),
        ("min_samples_leaf=20", dict(n_estimators=100, min_samples_leaf=20)),
        ("depth20+leaf10",    dict(n_estimators=100, max_depth=20, min_samples_leaf=10)),
    ]

    rows = []
    for name, kw in configs:
        t0 = time.time()
        clf = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, **kw
        )
        clf.fit(X_tr, y_tr)
        t_train = time.time() - t0

        P, R, F, tp, fp, fn = eval_on_validation(clf, data, val_ids)
        raw_mb, gz_mb = measure_size(clf)
        nodes = sum(t.tree_.node_count for t in clf.estimators_)
        depth = max(t.tree_.max_depth for t in clf.estimators_)

        rows.append((name, F, P, R, raw_mb, gz_mb, nodes, depth, t_train))
        print(f"[完成] {name:22s} F1={F:.4f} P={P:.4f} R={R:.4f} "
              f"raw={raw_mb:7.1f}MB gz={gz_mb:6.1f}MB 节点={nodes:,} 深度={depth} ({t_train:.0f}s)")

    print("\n" + "=" * 100)
    print(f"{'配置':<22} {'F1':>7} {'P':>7} {'R':>7} {'原始MB':>9} {'gzip MB':>9} {'压缩率':>7} {'节点数':>12} {'深度':>5}")
    print("-" * 100)
    base_f1 = rows[0][1]
    for name, F, P, R, raw_mb, gz_mb, nodes, depth, _ in rows:
        print(f"{name:<22} {F:>7.4f} {P:>7.4f} {R:>7.4f} {raw_mb:>9.1f} {gz_mb:>9.1f} "
              f"{raw_mb/gz_mb:>6.2f}x {nodes:>12,} {depth:>5}   (F1 {F-base_f1:+.4f})")


if __name__ == "__main__":
    main()
