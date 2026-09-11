"""
阶段2 · DataExplorer 类重构
============================
前置：黑马课 77-85 面向对象 + 86-87 异常 ✅
任务：把 阶段1_探索数据.py 的"过程式代码"改写成"面向对象"。

为什么用类？
  以后换数据源（华为官方数据 8 月中发布），只改类内部，
  外层调用代码不用动。这就是 OOP 的核心价值：把"一组相关操作"打包。

用法：
  1. 先把下面代码完整读一遍，每行都能讲出在干嘛
  2. 然后按文末"挑战任务"改代码
"""

import os

#from fontTools.misc.cython import returns
#from numpy.ma.core import append

# 数据目录：notebooks/ 的上一级 → data/UCI HAR Dataset/
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "UCI HAR Dataset")


class DataExplorer:
    """负责探索 UCI HAR 数据集的类。

    设计思路：
      - 数据目录在创建对象时传入，存在 self.data_dir
      - 读标签在 __init__ 里自动做，创建完对象标签就绪
      - 每个"能问的问题"是一个方法：标签分布 / 特征名
    """

    def __init__(self, data_dir):
        """创建对象时自动调用：存路径 + 读标签字典"""
        self.data_dir = data_dir
        self.labels = {}  # {0: 'WALKING', 1: 'WALKING_UPSTAIRS', ...}
        self.load_labels()  # 对象一出生，标签就读好

    def _read_lines(self, path):
        """
        打开文件加读取
        :return: 文件的所有行列表
        """
        path = os.path.join(self.data_dir, path)
        try:
            lines=[]
            with open(path) as f:
                for line in f:
                    lines.append(line)

                return lines
        except FileNotFoundError:
            print("没有找到文件的位置")
            return []

    def load_labels(self):
        """读 activity_labels.txt → self.labels

        用到了刚学的 try/except：文件不存在时不再直接崩溃，
        而是给出友好提示。（你之前跑代码遇到的 FileNotFoundError 就是它）
        """
        #path = os.path.join(self.data_dir, "activity_labels.txt")
        try:
            for line in self._read_lines( "activity_labels.txt"):
                idx, name = line.strip().split()
                self.labels[int(idx) - 1] = name
            print(f"✅ 标签读取成功: {len(self.labels)} 类")


        except FileNotFoundError:
            print(f"❌ 找不到文件: {"activity_labels.txt"}")
            print("   请检查 data/ 目录下有没有 UCI HAR Dataset 文件夹")

    def count_train_labels(self):
        """读 y_train.txt，统计每个活动的样本数 → 返回字典

        挑战 1：下面的 TODO 需要你补全（用普通字典，不用 Counter，
        因为 Counter 是还没学的语法——正好复习字典用法）
        """
        #path = os.path.join(self.data_dir, "train", "y_train.txt")
        counts = {}
        try:
            for line in self._read_lines("train/y_train.txt"):
               num = int(line.strip()) - 1  # 文件里 1-6，转成 0-5
                    # TODO: 如果 num 已经在 counts 里，counts[num] 加 1
               if num in counts:
                  counts[num] +=1
               else:
                  counts[num] = 1
                # TODO: 如果 num 不在，counts[num] = 1
                # 提示：用 if num in counts: ... else: ...
                # ← 把这两行 TODO 换成你的代码
            return counts




        except FileNotFoundError:
            print(f"❌ 找不到文件: {"train/y_train.txt"}")
            return {}

    def show_features(self, n=5):
        """打印前 n 个特征名"""
        #path = os.path.join(self.data_dir, "features.txt")
        try:
            for i, line in  enumerate(self._read_lines( "features.txt")):  # enumerate: 自动数行号
                if i >= n:
                    break
                print(f"  特征{i + 1}: {line.strip()}")


        except FileNotFoundError:
            print(f"❌ 找不到文件: {"features.txt"}")

    def summary(self):
        """汇总打印：标签 + 分布 + 特征，一行调用全搞定"""
        print("\n===== UCI HAR 数据概览 =====")
        print("活动标签:", self.labels)
        counts = self.count_train_labels()
        if counts:
            for num in sorted(counts):
                print(f"  活动 {num} ({self.labels.get(num, '?')}): {counts[num]} 个样本")
        self.show_features()
        print("============================\n")


if __name__ == "__main__":
    explorer = DataExplorer(DATA_DIR)
    explorer.summary()


"""
=====================================================================
挑战任务（按顺序做，做完一个发我一个）
=====================================================================

挑战 1（补代码）：把 count_train_labels 里的 TODO 补全。
  完成后运行，应该看到每个活动的样本数，加起来 = 7352。

挑战 2（理解题）：为什么 self.labels 在 __init__ 里就调用 load_labels()？
  如果把它去掉，后面 summary() 里 self.labels 会是空字典，为什么？

挑战 3（改造题）：现在 load_labels / count_train_labels / show_features
  三个方法都在做"打开文件 + 读行"这件重复的事。
  试试把"打开文件"抽成一个私有方法 _read_lines(self, filename)，
  返回文件的所有行列表，让上面三个方法都调用它。
  提示：私有方法就是名字前面加一个下划线 _，如 def _read_lines(self, path):
       这是"什么放一起、什么拆开"的第一次实战。

挑战 4（联想题）：等华为官方数据发布，你想换数据源。
  想一想：换数据时，你只需要改哪个方法的哪几行？哪些地方完全不用动？
"""
