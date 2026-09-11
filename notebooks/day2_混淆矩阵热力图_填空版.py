"""
Day 2 练习：混淆矩阵热力图（假数据版）
目标：画出 UCI HAR 6 分类结果的混淆矩阵热力图，看懂"模型错在哪"
运行：直接 python 跑，看到一张蓝色热力图即成功
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.f2py.cb_rules import cb_map
plt.rcParams['font.sans-serif'] = ['SimHei']
# ===== 假混淆矩阵：6 类活动 =====
# 行 = 真实类别，列 = 预测类别；数字 = 该(真实,预测)配对的样本数
# 对角线 = 判对的样本（越大越好），非对角线 = 误判
cm = np.array([
    [300,  20,  10,   0,   0,   0],   # 真实"1-走"：300 次判对，20 次误判成"上楼"...
    [ 15, 280,  25,   0,   0,   0],   # 真实"2-上楼"
    [ 10,  30, 290,   0,   0,   0],   # 真实"3-下楼"
    [  0,   0,   0, 350,  30,  20],   # 真实"4-坐"
    [  0,   0,   0,  20, 340,  30],   # 真实"5-站"
    [  0,   0,   0,  15,  25, 360],   # 真实"6-躺"
])
labels = ["1-走", "2-上楼", "3-下楼", "4-坐", "5-站", "6-躺"]

# ===== TODO 1: 用 plt.imshow() 画图（颜色用蓝色渐变 cmap="Blues"）=====
# 提示：plt.imshow(cm, cmap="Blues")
plt.imshow(cm,cmap="Blues")

#==== TODO 2: 加颜色条 =====
# 提示：plt.colorbar()
plt.colorbar(label='样本数')
# ===== TODO 3: 在每个格子里写上数字 =====
# 提示：两层 for 循环遍历 i(行), j(列)，用 plt.text(j, i, cm[i, j], ha="center", va="center")
# 想一想：为什么 x 坐标写 j、y 坐标写 i？
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
         plt.text(j,i,cm[i,j],ha='center',va='center')
# ===== TODO 4: 坐标轴装饰 =====
# 提示：plt.xticks(范围, labels) / plt.yticks(范围, labels)
#       plt.xlabel("预测类别") / plt.ylabel("真实类别") / plt.title("混淆矩阵")
#       plt.colorbar 的 label 可以写 "样本数"
plt.xticks(range(6),labels)
plt.yticks(range(6),labels)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
# ===== 自检：矩阵左上角 4×4 和右下角 2×2 颜色深（数字大）=====
# 你画的图，能看出"1-走/2-上楼/3-下楼"三兄弟互相误判较多吗？（看右上角的 20/15/10/30）

plt.show()  # 保持注释状态，画完再放开

