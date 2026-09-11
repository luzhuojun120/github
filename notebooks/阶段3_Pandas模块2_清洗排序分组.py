"""
阶段3 · 模块2：用 Pandas 在 UCI HAR 上做清洗/排序/分组
============================================================
前置：黑马课 145(清洗) / 146(排序) / 147(分组) ✅ 已学
任务：把模块1读出来的表，做点有意义的操作——清洗、排序、分组。
      学到的每个 Pandas 功能，立刻在真实数据上用一遍。

运行：直接跑本文件。每段输出前都有注释说明在干嘛。
"""

import os
import pandas as pd

os.chdir(os.path.dirname(__file__))
DATA_DIR = os.path.join("..", "data", "UCI HAR Dataset")

# 重用模块1的读取逻辑，先把表读进来
features = pd.read_csv(
    os.path.join(DATA_DIR, "features.txt"),
    sep=r"\s+", header=None, names=["序号", "特征名"],
)
seen = {}
clean_names = []
for name in features["特征名"]:
    if name in seen:
        seen[name] += 1
        clean_names.append(f"{name}_{seen[name]}")
    else:
        seen[name] = 1
        clean_names.append(name)

X_train = pd.read_csv(
    os.path.join(DATA_DIR, "train", "X_train.txt"),
    sep=r"\s+", header=None, names=clean_names,
)
y_train = pd.read_csv(
    os.path.join(DATA_DIR, "train", "y_train.txt"),
    header=None, names=["活动编号"],
)
df = X_train.copy()
df["活动编号"] = y_train["活动编号"]

# ============================================================
# 第 1 步：数据清洗（145 集）
# ============================================================

# 1.1 看看有没有缺失值（NaN）
# 561 列全看太多，只看几个示例列
print("=== 缺失值检查（仅前 3 列 + 活动编号）===")
print(df.iloc[:, list(range(3)) + [df.columns.get_loc("活动编号")]].isna().sum())

# 1.2 演示 fillna（实际上 UCI HAR 没缺失，这里只是让你看到方法）
# 如果有缺失值，可以填均值：
# df[col].fillna(df[col].mean(), inplace=True)

# 1.3 去重（演示：完全重复的行删除）
print("\n=== 去重前/后行数 ===")
print(f"去重前: {len(df)}")
df_clean = df.drop_duplicates()
print(f"去重后: {len(df_clean)}")   # 一样（UCI HAR 本身没重复样本）

# ============================================================
# 第 2 步：数据排序（146 集）
# ============================================================

# 2.1 按某个特征值排序，找最大/最小
# 找 "tBodyAcc-mean()-X"（第一个特征）值最大的 5 个样本
first_col = clean_names[0]
top5 = df.nlargest(5, first_col)   # nlargest = 按列排前 5 大
print("\n=== 第一特征值最大的 5 个样本 ===")
print(top5[[first_col, "活动编号"]])

# 2.2 多列排序：先按活动编号，再按第一特征
print("\n=== 按活动编号、第一特征排序前 10 行 ===")
#print(df.sort_values(["活动编号", first_col]).head(10)[["活动编号", first_col]])
print(df.sort_values(["活动编号", first_col]).head(10)[["活动编号"] +[clean_names[0], clean_names[280], clean_names[560]]])
# ============================================================
# 第 3 步：数据分组（147 集 · 重头戏）
# ============================================================

# 3.1 按活动编号分组，看每组的样本数（等价于模块1的 value_counts）
print("\n=== 每组样本数 ===")
print(df.groupby("活动编号").size())

# 3.2 按活动编号分组，算第一特征的均值和标准差
print("\n=== 每组在第一特征上的均值和标准差 ===")
print(df.groupby("活动编号")[first_col].agg(["mean", "std"]))

# 3.3 按活动编号分组，算多个关键特征的均值
# 选第一/中间/最后三个特征来演示
key_cols = [clean_names[0], clean_names[280], clean_names[560]]
print("\n=== 每组在三个关键特征上的均值（行=活动，列=特征）===")
print(df.groupby("活动编号")[key_cols].mean())

# ============================================================
# 检验：能看出不同活动的特征均值不一样 → 这就是分类器能"识别活动"的根源
# ============================================================
print("\n✅ 模块2 跑通！你已经用清洗/排序/分组把 UCI HAR 这张表玩起来了。")
print("   下一步（学完 148-150 Matplotlib 后）：解锁模块3，画图！")