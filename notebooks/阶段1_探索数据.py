"""
阶段1 · 探索 UCI HAR 数据集
============================
你刚学完：模块与包（73-76集）
黑马课进度：即将进入 OOP（77集）
对应竞赛：惯用手 IMU 进食检测

用法：直接在本文件运行即可。
"""

import os

# 把工作目录切到 notebooks/，相对路径就永远从这算起
os.chdir(os.path.dirname(__file__))

DATA_DIR = "../data/UCI HAR Dataset/"

# 1. 读活动标签字典
labels = {}
with open(DATA_DIR + "activity_labels.txt") as f:
    for line in f:
        idx, name = line.strip().split()
        labels[int(idx) - 1] = name
print("活动标签:", labels)

# 2. 训练集标签分布 —— 哪些活动样本多？
from collections import Counter
y_train = [int(line.strip()) for line in open(DATA_DIR + "train/y_train.txt")]
print("标签分布:", Counter(y_train))

# 3. 前 5 个特征名
print("前 5 个特征:")
for i, line in enumerate(open(DATA_DIR + "features.txt")):
    if i >= 5:
        break
    print(" ", line.strip())