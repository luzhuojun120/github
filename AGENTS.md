# AGENTS.md — 进食检测比赛

> 智能手表传感器进食检测算法项目（全国大学生生物医学工程创新设计竞赛 · 智能穿戴与运动健康技术赛道）

## 项目定位
基于智能手表 IMU/传感器数据的进食动作检测算法。定位：练手 + 考研作品集，目标院校深大/南方医 BME，目标企业联影。

## 怎么跑起来
- 数据集放 `data/`（已 gitignore，不提交）
- 算法代码写 `src/`，探索练习放 `notebooks/`
- 依赖清单：`requirements/requirements.txt`（baseline 已建，未提交）
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

## 当前状态（2026-09-10 更新 · 距 9-30 提交 20 天，其中 14 天在军训）
- ⚠️ **未报名** — 9-20 截止；指导老师未定（拟黄海辉，备选刘老师 / 汤凌冰）→ **9-11~9-13 是进军训前最后的窗口**
- ✅ **数据已全部到手**（2026-09-08）：50 个 sensorData zip + 288 条 mealinfo 标注（44 人）+ userinfo；重复副本 56 份已隔离待删
- ✅ **端到端管线跑通**（2026-09-10）：data_loader → preprocess → features → detect(RandomForest) → segment_meals → evaluate(IoU+F1)
- ❌ **PPG 非惯用手分支空白**：`src/detect_nondominant.py` 仍是 8-01 的 TODO 骨架，44 列 PPG 语义未破解
- `requirements/requirements.txt` 已建 baseline（numpy/pandas/scipy/scikit-learn/matplotlib）
- 黑马课路线：148-154 Matplotlib 保留，155-185 实战案例延后到军训后
- 📊 时间投入与阶段分析报告：`文件workbuddy/文件1/假菌_进食检测竞赛_时间投入与阶段分析_2026-09-10.html`（口径与优化空间见该文件）

## 已知缺陷（2026-09-10 只读诊断，未改动任何代码）
| 级别 | 问题 | 位置 |
|---|---|---|
| P0 | `from scipy.constants import precision` 会 ImportError | `src/evaluate.py` |
| P0 | 训练/测试按窗口随机切分，相邻重叠窗口泄漏 → 准确率不可信 | `src/detect.py` `train_test_split` |
| P0 | 只用单人单 zip：硬编码 `subject="HNU21026"`，官方给 40 人×3 天 | `scripts/build_dataset.py` |
| P1 | 只用 `ACC_X` 单通道，缺合幅值 mag 与陀螺仪 | `scripts/build_dataset.py` |
| P1 | 仅 5 个特征；主频未限 0.5–3 Hz 频段 | `src/features.py` |
| P1 | 标签"有交集即 1"，边缘窗口噪声大 | `src/preprocess.py` `make_labels` |
| P1 | 缺官方要求的高斯平滑 + 180s 合并 | `src/segment_meals.py` |
| P1 | 评估非一对一匹配，precision 虚高 | `src/evaluate.py` `evaluate` |
| P2 | `from notebooks.…import` 脏依赖 + sys.path typo(`workbuudy`) | `src/detect.py` |
| P2 | 采样率硬编码 105 Hz | `src/features.py` `ACC_FS` |

## 官方时间线（2026-09-06 核实）
- **2026-07-28**：赛道通知发布（华为终端有限公司支持）
- **2026-08-21**：官方数据已发布到华为云研究平台
- **2026-09-20**：线上报名截止
- **2026-09-30**：作品材料提交截止（源代码 + 设计报告）
- **2026-10 中旬**：决赛名单 → 线下答辩

## 数据集 schema（赛题 PDF V1.1）
- 采集：健康人群（**训练集 40 人 + 测试集 20 人**）；连续 3 天非睡眠时段全天候佩戴；每日更换佩戴手腕；单人 ≥6 条有效用餐记录
- 核心数据源：**PPG（光电容积描记）+ ACC（加速度计）**
- 三张表（数据库）：
  - `t_zsstnnrj_sensororiginaldata_system`：PPG + ACC 原始信号（externalId = 受试者唯一 ID）
  - `t_zsstnnrj_mealinfo_puaddqoq7`：用餐信息（**beforeTime / afterTime = 饭段起止**；dietaryType / drinkingVolume / satiety / foodName / tablewareType；dietaryHand = 惯用手，wearHand = 佩戴手）
  - `t_zsstnnrj_userinfobean_5d4l0nmp`：用户基本信息（ID / 年龄 / 性别 / 身高 / 体重）
- 下载：需签《数据使用与保密承诺书》发 bmedesign05@hainanu.edu.cn 后到 https://research.cloud.huawei.com/researchportal/#/dataTable

## 评测机制（赛题 PDF V1.1 原文 · 2026-09-06 全文核读）
- **两种佩戴场景，分别设计算法**（PDF 原话"本质差异，需分别设计针对性算法"）：
  - 惯用手佩戴：进食手部动作特征明显 → **IMU**（加速度计+陀螺仪）运动模式检测
  - 非惯用手佩戴：动作特征弱 → **PPG** 生理信号（心率变化等）检测
- **主指标**：检测出的进食事件与真值 **IoU > 0.25** 判为正确 → 算灵敏度(recall) + 阳性预测率(precision) → **F1 为最终评价标准**
- **次指标**：正确命中事件的**开始/结束时间判断误差 MAE**
- 关键后处理（`segment_meals` 参考 `meal_detection_heuristic`）：高斯平滑 → 阈值 → 边缘检测 → 180s 合并

## 评分标准（PDF 原文 · 重要！此前档案遗漏）
| 得分项 | 内容 | 分数 |
|---|---|---|
| 结果准确性 | 测试集预测结果客观评分（IoU>0.25 的 F1）| **50** |
| 先进性和创新性 | 设计思路创新、原理科学先进、创新点清晰 | **40** |
| 总结报告与答辩表现 | 报告条理 + 答辩 PPT/陈述 | **10** |

## 提交要求（PDF 原文）
- 提交物：**最终算法代码（Python）+ 总结报告 + 可执行文件**（运行后生成测试集结果）
- 可执行文件无法运行 → **结果准确性 0 分**
- 训练代码 + readme（复现说明）备查，无法复现取消资格
- **官网仅能提交一次，提交后无法更改**——确认最终版再交
- ⚠️ **预测试机制**：初赛截止前一周（≈9/23）可将可执行文件发 bmedesign03@hainanu.edu.cn（文件命名 "answer_队伍名_赛题_提交日期"，邮件主题 "可执行文件测试_队伍名_赛题_提交日期"）→ 组委会测试给参考分 → 可迭代改进后再官网提交
- 报告：正文前言+引入 ≤2 页、方案/结果/讨论 ≤8 页、整体 ≤12 页
- 附件文档待取：测试接口说明（决定可执行文件怎么写）+ 异常数据说明

## 数据（PDF 原文 + 官网公告）
- 信号：**IMU（加速度计 + 陀螺仪）+ PPG**（华为手表采集，训练集 40 人 + 测试集 20 人，连续 3 天）
- 官网 2026-08-21 公告数据已发布；PDF 注"数据获取方法后续另行通知，报名后关注"
- 平台：https://research.cloud.huawei.com.cn/researchportal/#/dataTable

## 数据下载状态（2026-09-07 实测）
- 附件工具包已解压至 `进食检测比赛/华为数据下载工具/`（含下载脚本 + HiResearch SDK whl + DataTable.txt）
- SDK 已装到 envs/default（需 setuptools 补 distutils，Python 3.12+ 已移除）
- ⚠️ **下载被华为平台权限挡住**：所有查询返回 `HTTP 500 {"code":16001,"message":"User role does not permission to access"}`（匿名和带凭证都如此）
- 最可能根因：**数据访问权与报名/平台授权绑定**（PDF 原话"报名后关注数据获取"）；用户 9-20 报名截止前未报名
- 排查路径：① 确认报名（含指导教师）② 网页登录 research 平台核对凭证/项目 ③ 联系组委会 bmedesign03@hainanu.edu.cn 附 16001 报错
- 备选：DataTable.txt 已含两张表字段（sensor 表：externalid/sensorData/timeStamp；mealinfo 表 `t_zsstnnrj_mealinfo_puadqog7`：externalid/dietaryType/beforeTime/afterTime/.../wearHand）——注意 mealinfo 表名是 **puadqog7**（非 PDF 截图手抄的 puaddqoq7，已核对附件原文）

## 数据下载完成（2026-09-08 · ✅ 核心数据到手）
- **sensorData 原始信号 zip × 50**（PPG+ACC，~1GB）→ `data/raw/sensorData/`（✅ 全齐）
- **mealinfo 标注表 × 288 条**（44 人；午餐117/晚餐129/早餐27/加餐15；含 externalid/dietaryType/beforeTime/afterTime/foodName/wearHand）→ `data/raw/mealinfo_标注表.csv`（✅ 已导出）
- **diningPictures 用餐照片 × 47**（差 3，非训练必需，优先级低）→ `data/raw/diningPictures/`
- 重复副本 56 份已隔离 → `data/raw/_重复副本_待删/`（确认后可清，~400MB）
- 下载方式：**网页表内逐行下载按钮**（无批量；手动点出）；下载目录 `E:\浏览器下载` 已清空
- 关键认知：①"主表 vs copy 表"数据相同，别在 copy 重复下 ② 网页"导出"是表复制不是文件下载 ③ SDK 旧版 batch_download_file 卡 tableId，新版 hiresearchsdk 才有 get_attachment（未获取）
- UserInfoBean（受试者身高/体重/年龄）表尚未导出（可选补充）
- 下一步：按 `data/raw/mealinfo_标注表.csv` 的 beforeTime/afterTime 对齐 sensorData zip 内时间戳 → 训练窗口打标签

## 数据申请官方流程（2026-09-07 补充 · 来源：夸克扫描王_数据使用说明(1).pdf，浏览器下载目录）
- 流程：① 承诺书末尾填参赛队信息 → ② 负责人手写签名 → ③ 扫描件 PDF **+ 华为创新研究网站 ID** → 发 bmedesign05@hainanu.edu.cn → ④ **邮件标题必须 = 参赛队名 + "赛题名" + "数据使用申请"**
- 用户 14:41 已发首封邮件（格式不符但**已触发授权**：组委会有回信，凭账号 python2026 已能进"赛题二训练集"项目并浏览/导出数据表）
- 浏览器实际下载入口（2026-09-07 21:37 实测）：**左侧菜单"数据管理"→ "数据表"** → 列出所有表（MealInfo、传感器原始数据=SensorOriginalData、UserInfoBean 等）→ **点表名进入"Table列表"页** → 顶部按钮 **"选择字段 / 设置过滤条件 / 导出 / 导出记录"** → 设时间窗 2026-06-20 ~ 2026-08-13 → 导出 CSV
- 实战分工：**MealInfo 表（几百条）走网页导出**；**SensorOriginalData 表（千万行）走 SDK**——SDK 路线：网页"我的凭证"生成 access key → 替换下载脚本 429-430 行的两个值 → 跑
- 用户账号**已授权可用**（7-31 ~ 8-02 已有 10 条 MealInfo 记录可浏览），**不必再纠结补发邮件**

## 下一步（2026-09-10 重排 · 9-11~9-13 只剩 3 个完整工作日）
- 已完成：2 数据下载 / 3 data_loader / 4 preprocess / 5 特征工程(v0.1) / 6 baseline / 7 segment_meals(v0.1) / 8 evaluate(v0.1)

1. **报名（9-20 前，最高优先级）** — 9-11 上午发邮件找指导老师；没报名代码再好也是 0 分
2. **修 P0 三处**：evaluate 的 import、切分改 subject-wise、build_dataset 批量化到 40 人
3. **提特征**：加合幅值 mag + 过零率/RMS/峰值计数；主频限 0.5-3 Hz
4. **补后处理**：高斯平滑 + 阈值 + 180s 合并（官方明确要求）
5. **评估改一对一匹配**，拿到可信 F1
6. **设计报告初稿**（创新性 40 分 + 报告 10 分，全靠它）— 军训期间主攻
7. **PPG 分支**：推进队友鲁焯俊负责的非惯用手场景，或自己破解 44 列语义
8. **预测试**：≈9-23 前把可执行文件发 bmedesign03@hainanu.edu.cn 拿参考分
