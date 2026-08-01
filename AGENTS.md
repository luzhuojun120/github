# AGENTS.md — 进食检测比赛

> 智能手表传感器进食检测算法项目（全国大学生生物医学工程创新设计竞赛 · 智能穿戴与运动健康技术赛道）

## 项目定位
基于智能手表 IMU/传感器数据的进食动作检测算法。定位：练手 + 考研作品集，目标院校深大/南方医 BME，目标企业联影。

## 怎么跑起来
- 数据集放 `data/`（已 gitignore，不提交）
- 算法代码写 `src/`，探索练习放 `notebooks/`
- 依赖清单：`requirements/requirements.txt`（待建）
- 运行环境：Python 3.13

## 技术栈
Python · （待定：sklearn / PyTorch 视数据集规模与任务而定）

## 目录约定
| 路径 | 用途 |
|------|------|
| `data/` | 数据集，本地保留不上传 |
| `src/` | 算法代码 |
| `notebooks/` | 探索性练习 |
| `requirements/` | 依赖清单 |

## 当前状态（2026-08-01）
- git 仓库已建，`.gitignore` 已配（忽略 `.idea/`、`data/`、`__pycache__`、模型大文件等）
- 目录骨架就位，尚无算法代码
- 远程：`github.com/luzhuojun120/github.git`（push 因网络被重置，待重试）
- 下一步：确定数据集来源，在 `src/` 写数据预处理与进食检测原型
