"""paths.py —— 全项目统一的路径基准。

【为什么要有这个文件】
绝对路径（如 E:\\workbuddy\\...）在别人机器上不存在，打包成 exe 后更会失效。
所有脚本统一从这里拿路径，就只需要维护这一处。

【两种运行形态】
    直接跑 .py   →  __file__ 是脚本真实路径        → BASE = 项目根目录
    打包成 exe   →  __file__ 是临时解压目录（会变） → BASE = exe 所在目录
    所以打包后，请把 exe 放在「与 data/ 同级」的位置，或改用命令行参数指定数据目录。

【用法】
    from paths import BASE, DATA, RAW, CACHE, SENSOR
    p = os.path.join(RAW, "sensorData", "sensorData-xxx.zip")
不要在别的文件里再写任何盘符路径。
"""
import os
import sys

if getattr(sys, "frozen", False):
    # 已打包：__file__ 不可信，用 exe 的真实位置
    BASE = os.path.dirname(os.path.abspath(sys.executable))
else:
    # 直接跑 .py：本文件在 src/ 下，上一级就是项目根
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, "frozen", False) else BASE

DATA = os.path.join(BASE, "data")
RAW = os.path.join(DATA, "raw")
CACHE = os.path.join(DATA, "cache")
SENSOR = os.path.join(RAW, "sensorData")

# 训练产物 / 交付产物（B-2 起使用）
# gzip 压缩的 pickle：实测 546 MB → 118 MB（4.6x），加载时用 gzip.open + pickle.load
#
# 查找顺序（打包后两种位置都要兼容）：
#   1) exe 同目录 —— 评委若想换模型，把新 model.pkl.gz 放在 exe 旁边即可覆盖
#   2) _MEIPASS  —— 打包时内嵌的那份（PyInstaller 把 datas 放在 _internal/ 下）
# 前者优先，找不到才回退，这样"内嵌可用"和"现场可替换"两个需求都满足。
_MODEL_CANDIDATES = [os.path.join(BASE, "model.pkl.gz")]
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _MODEL_CANDIDATES.append(os.path.join(sys._MEIPASS, "model.pkl.gz"))

_model_path = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)), _MODEL_CANDIDATES[0])
MODEL = _model_path


if __name__ == "__main__":
    for name in ["BASE", "SRC", "DATA", "RAW", "CACHE", "SENSOR", "MODEL"]:
        p = globals()[name]
        print(f"{name:>7} = {p}    {'[存在]' if os.path.exists(p) else '[不存在]'}")
