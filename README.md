# competition_core —— 非惯用手进食检测核心

本目录提供基于 ACC + GYRO 六轴特征的非惯用手进食事件检测。随机森林在
500 点窗口上分类，每次移动 250 点；华为数据约 100 Hz 时相当于 5 秒窗口、
2.5 秒步长。

## 文件说明

| 文件 | 作用 |
|---|---|
| `detect_nondominant.py` | 非惯用手窗口检测、整段事件检测和流式检测 |
| `random_forest.py` | 六轴特征提取、随机森林训练和模型读写 |
| `train_model.py` | 从标注餐次与映射表训练随机森林 |
| `data_loader.py` | 读取华为 sensorData 压缩包，兼容 npy/npz/csv/txt 单列信号 |
| `preprocess.py` | 滑窗、归一化、运动特征、事件平滑与合并 |
| `metrics.py` | 事件级 IoU、一对一匹配、灵敏度、PPV 和 F1 |
| `__init__.py` | 包入口，导出事件评估接口 |
| `run_detection.py` | 非惯用手命令行检测入口 |
| `evaluate_robustness.py` | 合成信号检查，不代表真实数据准确率 |

## 创新点实现

## 四层方案入口

`pipeline.py` 提供与竞赛方案对应的四层架构：

1. **DataLayer**：按 ACC 时间戳切分连续片段，将 GYRO/PPG 插值到统一时钟，处理缺失值并做轻量降噪。
2. **FeatureLayer**：以 10 秒窗口、5 秒步长提取 IMU 时域统计、PPG 周期性、频带能量、jerk、静止比例等特征。
3. **ModelLayer**：使用分类器输出逐窗口进食概率，并融合可配置的进食时段先验。
4. **DecisionLayer**：对概率做平滑，以进入/退出双阈值状态机解码事件，合并短间隔并剔除过短片段。

示例：

```python
from competition_core.pipeline import EatingPipeline, PipelineConfig

pipeline = EatingPipeline(PipelineConfig(min_duration_s=60))
pipeline.fit([(training_bundle, [(start_s, end_s)])])
events = pipeline.predict(test_bundle)
report = pipeline.evaluate(test_bundle, truth_events)
```

该入口与原有 `random_forest.py`、`detect_nondominant.py` 接口并存，便于替换模型和做消融实验。

## 对外 JSON 接口

启动本地服务：

```bash
python -m competition_core_stitched.api --host 127.0.0.1 --port 8080
```

健康检查：`GET /health`

检测：`POST /v1/detect`

```json
{
  "sensorData": ["数据/sensorData/a.zip", "数据/sensorData/b.zip"],
  "model": "meal_random_forest.joblib",
  "stitch": true,
  "minDurationS": 300,
  "hourRange": "5,24"
}
```

训练：`POST /v1/train`

```json
{
  "zipDir": "数据/sensorData",
  "mealinfo": "数据/mealinfo_标注表.csv",
  "mapping": "数据/sensor_下载映射表.csv",
  "model": "meal_random_forest.joblib"
}
```

评估：`POST /v1/evaluate`

```json
{
  "zipDir": "数据/sensorData",
  "mealinfo": "数据/mealinfo_标注表.csv",
  "mapping": "数据/sensor_下载映射表.csv",
  "externalid": "HNU21026",
  "model": "meal_random_forest.joblib",
  "stitch": true
}
```

接口返回 JSON。检测事件统一返回 `start`、`end`、`duration_s`，并同时返回相对文件起点的 `relative_start`、`relative_end`。字段名对应附件中的 `sensorData`、`externalid`、`beforeTime`、`afterTime`、`dietaryHand` 和 `wearHand`。

1. **边界膨胀量化**：`boundary_expansion_iou(L, delta)` 按截图中的
   `IoU = L / (L + delta)` 量化事件长度为 `L`、总边界膨胀量为 `delta` 时的重叠率；
   `boundary_delta_from_iou` 可由目标 IoU 反算允许的边界偏差。此模型用于边界误差分析，
   竞赛事件匹配仍使用真实区间 IoU 和严格阈值。
2. **断续采集切分**：读取器保留 ACC/PPG 原始时间戳。ACC 相邻采样时间间隔超过
   `max_gap_s`（默认 60 秒）时，检测器分段处理，不跨缺口做平滑和事件合并。
3. **跨文件数据组织**：`detect_nondominant_files(paths, stitch=True)` 可接收 ZIP 文件列表或目录，
   按 ACC 真实时间排序，将多个导出文件的窗口统一平滑；文件之间超过 `max_gap_s` 的缺口不会
   被填充，也不会跨缺口合并事件。默认 `stitch=False` 仍按文件返回结果。

## 真实数据评估

先使用独立训练数据生成模型。映射表用于将 ZIP 文件关联到受试者，避免把其他受试者
同一时间的餐次误当成当前 ZIP 的标签：

```bash
python -m competition_core.train_model --zip-dir "训练数据/sensorData" \
  --meals "训练数据/mealinfo_标注表.csv" --mapping "训练数据/sensor_下载映射表.csv" \
  --output "meal_random_forest.joblib"
```

用未参与训练的受试者或 ZIP 做评估，程序执行严格 IoU 一对一匹配，并输出命中、漏检、
边界 MAE 和真实/预测事件对照图：

```bash
python -m competition_core.evaluate_real_data --zip-dir "数据/sensorData" \
  --meals "数据/mealinfo_标注表.csv" --mapping "数据/sensor_下载映射表.csv" \
  --participant HNU21026 --model "meal_random_forest.joblib" \
  --output "evaluation.json" --plot "meal_comparison.png"
```

边界 MAE 是所有匹配事件的起点和终点绝对误差的平均值，单位为秒。无匹配事件时值为
`null`。应按受试者划分训练与验证，避免同一受试者同时进入两侧。

## 使用方法

在本目录运行：

```bash
python run_detection.py --zip "真实数据路径/sensorData-xxx.zip" --model "meal_random_forest.joblib"
```

多个连续传感器导出文件（例如 35 个 ZIP）可一次拼接检测：

```bash
python run_detection.py --zip "真实数据路径/队友_开工包_HNU21026" \
  --model "meal_random_forest.joblib" --stitch-files
```

在本目录的上级目录运行：

```bash
python -m competition_core.run_detection --zip "真实数据路径/sensorData-xxx.zip" --model "meal_random_forest.joblib"
```

默认运行非惯用手检测。兼容旧参数 `--mode nondominant`。每次推理必须指定训练好的模型。
命令行可接收一个或多个 ZIP 文件，或包含 ZIP 的目录。默认输出每个文件相对第一条
ACC 数据的 `start_s,end_s`；指定 `--stitch-files` 后输出拼接结果相对整个文件集最早
ACC 数据的时间，单位为秒。

支持 `--hour-range 5,24`（UTC+8，默认）、`--hour-range none`（不限时段）、
`--min-duration 300`（最短事件秒数）及 `--threshold 0.5`（模型概率阈值）。

## Python 接口与评估

```python
from competition_core.data_loader import read_sensor_zip
from competition_core.detect_nondominant import detect_nondominant_events
from competition_core.metrics import compute_f1
from competition_core.random_forest import load_model

bundle = read_sensor_zip("真实数据路径/sensorData-xxx.zip")
model = load_model("meal_random_forest.joblib")
predicted = detect_nondominant_events(bundle, model=model)
# Python 检测接口返回 Unix 时间戳，单位为秒。
# 示例标注为 HNU21026 的左手佩戴、右手进食片段；应加载覆盖该餐的压缩包。
truth = [(1785382231.0, 1785382971.0)]
print(compute_f1(predicted, truth, iou_threshold=0.25))
```

真实标注的 `beforeTime` / `afterTime` 是毫秒，评估前须除以 1000。
只比较同一受试者、同一时间范围的事件；不能将命令行相对时间直接与绝对时间标注比较。
严格使用 `IoU > 0.25`，通过最大匹配实现一对一配对，再计算 TP、FP、FN、
灵敏度、PPV 和 F1。两组事件均为空时，现有评估函数约定 F1 为 1。

## 数据与算法限制

华为压缩包必须同时含有效 ACC 和 GYRO 三轴数据。六轴窗口特征包含统计量；采集间隔
超过 60 秒时会分段，事件平滑不会跨越缺口。训练 ZIP 必须能由映射 CSV 关联到受试者。
训练样本覆盖面和受试者划分会直接影响泛化能力；请在独立验证数据上确定分类阈值。

依赖：Python ≥3.10、numpy、scikit-learn、joblib、matplotlib。
