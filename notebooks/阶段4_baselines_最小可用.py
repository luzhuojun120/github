"""
阶段4 · baselines 最小可用版（7 天 deadline 紧急版）
====================================================
任务窗口：2026-09-06 → 2026-09-12 开学前 7 天
目标：跑通端到端分类流水线（数据 → 模型 → 预测 → 指标），作为作品 baselines。

策略：跳过黑马课 148-185（Matplotlib + 实战案例对 baselines 无贡献），
      延后到军训后补，不算跳课。
本文件先用 UCI HAR 数据跑通，等团队数据 ready 后改 5 处即可。

输出：
  - 控制台 accuracy / precision / recall / F1
  - 控制台混淆矩阵（看哪些类被搞混）
  - predictions_baseline.csv（提交用预测文件）

跑法：直接 python 阶段4_baselines_最小可用.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

os.chdir(os.path.dirname(__file__))
DATA_DIR = os.path.join("..", "data", "UCI HAR Dataset")

# === 第 ① 处：读数据（复用模块 1 的去重模式）===
features = pd.read_csv(
    os.path.join(DATA_DIR, "features.txt"),
    sep=r"\s+", header=None, names=["序号", "特征名"],
)
seen = {}; clean_names = []
for name in features["特征名"]:
    if name in seen:
        seen[name] += 1
        clean_names.append(f"{name}_{seen[name]}")
    else:
        seen[name] = 1
        clean_names.append(name)

X = pd.read_csv(
    os.path.join(DATA_DIR, "train", "X_train.txt"),
    sep=r"\s+", header=None, names=clean_names,
)
y = pd.read_csv(
    os.path.join(DATA_DIR, "train", "y_train.txt"),
    header=None, names=["活动编号"],
)

print(f"数据规模: {X.shape[0]} 行 × {X.shape[1]} 特征")
print(f"标签分布: {y['活动编号'].value_counts().sort_index().to_dict()}")

# === 第 ② 处：切分训练/测试集（stratify 保证各类比例一致）===
X_train, X_test, y_train, y_test = train_test_split(
    X, y["活动编号"], test_size=0.2, random_state=42, stratify=y["活动编号"],
)
print(f"\n训练集 {X_train.shape[0]} 行, 测试集 {X_test.shape[0]} 行")

# === 第 ③ 处：Baseline A — RandomForest（万金油，默认首选）===
print("\n" + "=" * 60)
print("Baseline A: RandomForestClassifier")
print("=" * 60)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nAccuracy: {acc_rf:.4f}")
print(classification_report(y_test, y_pred_rf, digits=4))

# === 第 ④ 处：Baseline B — LogisticRegression（线性模型，对比参照）===
print("\n" + "=" * 60)
print("Baseline B: LogisticRegression（线性模型）")
print("=" * 60)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nAccuracy: {acc_lr:.4f}")
print(classification_report(y_test, y_pred_lr, digits=4))

# === 第 ⑤ 处：混淆矩阵（看哪些类被搞混，方便调优）===
print("\n" + "=" * 60)
print("混淆矩阵（RandomForest，6 个活动）")
print("=" * 60)
cm = confusion_matrix(y_test, y_pred_rf)
labels = sorted(y["活动编号"].unique())
print(pd.DataFrame(cm, index=labels, columns=labels))

# === 输出预测文件（提交用）===
sub = pd.DataFrame({
    "活动编号_真实": y_test.values,
    "活动编号_预测": y_pred_rf,
})
sub.to_csv("predictions_baseline.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ 已输出 predictions_baseline.csv（{len(sub)} 行）")

# === 总结 ===
print("\n" + "=" * 60)
print("Baselines 总结")
print("=" * 60)
print(f"  RandomForest:        {acc_rf:.4f}")
print(f"  LogisticRegression:  {acc_lr:.4f}")
print("\n下一步：把这份脚本换数据源，跑你们进食检测项目的版本")
print("改的位置：第 ①②③④⑤ 处 + DATA_DIR 路径")