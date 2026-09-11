"""
UCI HAR 练手指南 · 假菌专用
============================
数据集：UCI HAR Dataset（人类活动识别，手机 IMU 数据）
路径：../data/UCI HAR Dataset/
对应赛题：惯用手 IMU 进食检测

这个文件的用法：不是一次性全跑，是"你的黑马课学到哪、你就往下做到哪"。
每一步都标了前置依赖的黑马课章节。

=========================================================================
数据集概览（先读一遍，不用写代码）
=========================================================================

六个活动（你要分类的标签）：
  0 WALKING          走路
  1 WALKING_UPSTAIRS 上楼梯
  2 WALKING_DOWNSTAIRS 下楼梯
  3 SITTING          坐着
  4 STANDING         站着
  5 LAYING           躺着

数据已经在 train/ 和 test/ 下拆好了：
  - train/X_train.txt  → 7352 个样本 × 561 个特征  (70%）
  - train/y_train.txt  → 7352 个标签（0~5）
  - test/X_test.txt    → 2947 个样本 × 561 个特征  (30%）
  - test/y_test.txt    → 2947 个标签

特征（561 个）是原作者帮你算好的时域/频域统计量，不用你自己写滑窗。
变量名在 features.txt 里（如 tBodyAcc-mean()-X 表示"时域·身体加速度·X轴·均值"）。

Inertial Signals/ 里的原始加速度/陀螺仪序列，等你学到 scipy 滤波时再用。
现在先用 X_train/X_test 上手。

=========================================================================
阶段 1：模块与包 —— 把数据读进来
黑马课前置：模块与包（第73-75集）
=========================================================================

目标：会用 import 导入 numpy/pandas，把 txt 文件读成程序能处理的格式。
"""

# ----- 第 1 步：读 X_train.txt -----
# 提示：np.loadtxt() 直接读空格分隔的 txt
import numpy as np

DATA_DIR = "../data/UCI HAR Dataset/"

# TODO：补全下面两行
# X_train = np.loadtxt(DATA_DIR + "???")
# y_train = np.loadtxt(DATA_DIR + "???")

# 读完后，print(X_train.shape) 应该输出 (7352, 561)
# 读完后，print(y_train.shape) 应该输出 (7352,)

"""
# ----- 第 2 步：读标签名 -----
# activity_labels.txt 里是 1 WALKING 这样的格式
# 用 Python 原生 open() + readlines() 就行

# TODO：把 activity_labels.txt 读到字典里
# labels = {}
# for line in open(DATA_DIR + "activity_labels.txt"):
#     数字, 名字 = line.strip().split()
#     labels[int(数字)] = 名字
# print(labels)
"""


# =========================================================================
# 阶段 2：numpy 基础 —— 理解你手里的数据长什么样
# 黑马课前置：numpy 基础（第137集起）
# =========================================================================

"""
目标：用 numpy 的 shape / 切片 / 统计方法，回答以下几个问题。

1. X_train 有多少行、多少列？  → X_train.shape
2. 负样本有多少？正样本有多少？  → np.unique(y_train, return_counts=True)
3. 第一列的均值是多少？标准差是多少？ → np.mean, np.std
4. 前 50 个样本的特征 1-3 是什么？  → X_train[:50, :3]

----- 练习：回答上面 4 个问题（用注释写答案）-----

# 你的代码：
"""


# =========================================================================
# 阶段 3：sklearn 入门 —— 跑第一个分类器
# 黑马课前置：sklearn（机器学习，约第145集后）
# =========================================================================

"""
目标：用一个最简单的分类器，从 561 个特征 → 6 类活动标签。

整体流程：
  X_train, y_train  →  fit() 学习  →  predict(X_test)  →  和 y_test 比较算准确率

----- 第 3 步：你的第一个机器学习模型 -----
"""

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# TODO：补全
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.???(X_train, y_train)          # 训练
# y_pred = clf.???(X_test)           # 预测
# acc = accuracy_score(y_test, y_pred)  # 准确率
# print(f"准确率: {acc:.4f}")         # 期望 > 90%
# print(classification_report(y_test, y_pred, target_names=labels.values()))

"""
准确率 > 90% 就够了，UCI HAR 是成熟数据集，分类不难。
你练的是"整个流程走通"，不是冲高分。
"""


# =========================================================================
# 阶段 4：matplotlib —— 画混淆矩阵
# 黑马课前置：matplotlib（约第137集后，可以和第 3 阶段并列做）
# =========================================================================

"""
目标：用热力图看哪些活动容易搞混。

----- 第 4 步：画混淆矩阵 -----
"""

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns  # 可选，用 plt.imshow 也行

# cm = confusion_matrix(y_test, y_pred)
# 画热力图，看看 WALKING_UPSTAIRS 和 WALKING_DOWNSTAIRS 是不是容易搞混
# （直觉上这俩应该是互相误判最多的）


# =========================================================================
# 阶段 5：连接回赛题 —— 从"活动分类"到"进食检测"
# 完成前面 4 阶段后做
# =========================================================================

"""
UCI HAR 的分类是"走路/上楼/坐/站/躺"，赛题是"吃/没吃"。
核心流程完全一样：

  原始信号 → 滑窗 → 提特征 → 分类器 → 每个窗口输出 0/1

区别：
  - UCI HAR：6 类，帮你拆好了特征
  - 赛题：2 类（吃/没吃），你要自己写滑窗 + 提特征

所以你在 UCI HAR 上练熟的"读数据→fit→predict→评估"这条流水线，
就是赛题惯用手分支的蓝本。等你写完这 4 个阶段再回来看这句，你会懂的。
"""


# =========================================================================
# 进度追踪
# =========================================================================
"""
每完成一个阶段，在下面打勾：

[ ] 阶段 1：数据读进来了（print shape 验证过）
[ ] 阶段 2：用 numpy 回答完 4 个问题
[ ] 阶段 3：RandomForest 跑出 >90% 准确率
[ ] 阶段 4：画出混淆矩阵
[ ] 阶段 5：用自己的话写两句话，说 UCI HAR 流程和赛题的相同点/不同点
"""
