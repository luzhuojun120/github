# 进食检测比赛项目

全国大学生生物医学工程创新设计竞赛 · 智能穿戴与运动健康技术赛道

## 赛题
基于智能手表传感器的进食检测算法：用智能手表的 IMU（加速度计 + 陀螺仪）+ PPG 信号，自动检测佩戴者是否在进食，并给出每次进食的**开始与结束时间**。评测指标为**事件级 IoU（>0.25 判对）+ F1**。

## 项目结构
- `data/`：数据集（不上传 GitHub，已 gitignore）
  - `data/raw/sensorData/`：传感器原始信号 zip（每 zip 一段，约 1.2–17 小时）
  - `data/raw/mealinfo_标注表.csv`：进食标注（`externalid / beforeTime / afterTime / dietaryHand / wearHand / …`）
  - `data/raw/sensor_下载映射表.csv`：受试者 ↔ zip 映射（含各 zip 的时间区间）
- `src/`：算法模块
  - `data_loader.py` 读传感器 zip → DataFrame（列：`ACC_TIME, ACC_X/Y/Z, GYRO_X/Y/Z`）
  - `preprocess.py` 滑窗 `sliding_window` / 打标签 `make_labels`
  - `features.py` 窗口 → 5 维特征（mean/std/max/min/主频）
  - `detect.py` 窗口级分类（RandomForest）
  - `detect_dominant.py` **惯用手分支（IMU 6 轴 → 30 维特征）**，主责：用户
  - `detect_nondominant.py` 非惯用手分支（PPG 为主），主责：队友
  - `segment_meals.py` 窗口预测 → 饭段（起止时刻）
  - `evaluate.py` 事件级 IoU / precision / recall / F1
- `scripts/`：入口脚本（`build_dataset.py` 串联数据管线）
- `notebooks/`：探索性练习
- `requirements/`：依赖清单

## 怎么跑
```bash
pip install -r requirements/requirements.txt

# 单模块自测（每个 src 模块末尾都有 __main__ 自测块）
python src/features.py
python src/segment_meals.py
python src/evaluate.py

# 串联管线
python scripts/build_dataset.py
```

## 环境
Python 3.13 · numpy / pandas / scipy / scikit-learn / matplotlib

## 当前状态与已知缺陷
见 [`AGENTS.md`](AGENTS.md)（含最新进展、缺陷清单与下一步）。
技术发现与报告素材见 `../文件workbuddy/文件1/进食检测_技术发现与报告素材_假菌.md`。
