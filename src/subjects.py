"""
subjects.py
===========
受试者名册模块（参赛正式代码 v0.1 · 2026-09-23 新建）

作用：
    把三张原始表（mealinfo 标注表 / sensor_下载映射表 / userinfo 用户信息表）
    整理成一份「干净的名册」——谁有效、他有哪些饭段、对应哪些 zip。
    供 dataset 模块遍历使用，避免筛选逻辑散落各处。

为什么需要它：
    原始表存在三类脏数据（2026-09-23 实测）：
      ① ID 格式不统一：`HNU 21013`(带空格) / `hnu21033`(小写) / `HNU212020`(位数异常)
      ② 对不上的行：`AC3`(有标注无 zip) / `6C7`、`NAN`(有 zip 无标注)
      ③ ⚠️ 相似但不同人：`HNU21026` 与 `HNU21026J` 是两台手表(MAC 不同)的两个人

    → 规范化必须只处理「空白/横线/下划线」和大小写，**绝不能碰字母数字**，
      否则 HNU21026J 会被并进 HNU21026，造成数据污染。

模块内函数：
    norm_id(s)                      单个 ID 规范化
    load_meal_table(raw_dir)        读标注表 + 规范化 + 加 is_dominant 列
    load_zip_table(raw_dir)         读映射表 + 规范化
    load_subjects(raw_dir)          构建有效受试者名册（dict）

用法示例：
    from subjects import load_subjects
    subjects = load_subjects(r"...\\data\\raw")
    for sid, info in subjects.items():
        print(sid, len(info["meals"]), len(info["zips"]))

作者：品食Panda 队 · 假菌
"""

import os
import re

import pandas as pd


MEAL_FILE = "mealinfo_标注表.csv"
ZIP_FILE = "sensor_下载映射表.csv"
USER_FILE = "userinfo_用户信息.csv"


def norm_id(s):
    """
    ID 规范化：去掉空白/横线/下划线，转大写。

    ⚠️ 只做这些，绝不删字母数字——`HNU21026J` 必须保持独立（它是另一个人）。

    参数
    ----
    s : str
        原始 externalid，如 "HNU 21013" / "hnu21033"

    返回
    ----
    str，如 "HNU21013" / "HNU21033"
    """
    return re.sub(r"[\s\-_]", "", str(s).strip().upper())


def load_meal_table(raw_dir):
    """
    读 mealinfo 标注表，规范化 ID，并加一列 is_dominant。

    is_dominant 判据：wearHand.startswith(dietaryHand)
        True  → 表戴在吃饭那只手上 = 惯用手场景（本项目分支）
        False → 表戴在另一只手上   = 非惯用手场景（队友分支）

    返回
    ----
    DataFrame，列在原表基础上增加：id_n（规范化ID）、is_dominant（bool）
    """
    path = os.path.join(raw_dir, MEAL_FILE)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["id_n"] = df["externalid"].map(norm_id)
    df["is_dominant"] = df.apply(
        lambda r: str(r["wearHand"]).startswith(str(r["dietaryHand"])), axis=1
    )
    return df


def load_zip_table(raw_dir):
    """
    读 sensor_下载映射表，规范化 ID。

    返回
    ----
    DataFrame，列在原表基础上增加：id_n
    """
    path = os.path.join(raw_dir, ZIP_FILE)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["id_n"] = df["externalid"].map(norm_id)
    return df


def load_subjects(raw_dir, dominant_only=False):
    """
    构建有效受试者名册：标注表 ∩ 映射表 都有的 ID。

    参数
    ----
    raw_dir : str   原始数据目录
    dominant_only : bool
        True  → 只保留有「惯用手饭段」的受试者（本项目分支用）
        False → 全部有效受试者

    返回
    ----
    dict，结构：
        {
          "HNU21026": {
              "meals": [(beforeTime, afterTime), ...],
              "meal_rows": DataFrame 该人的标注行（含 dietartType/is_dominant 等）,
              "zips": ["sensorData-xxx.zip", ...],
              "zip_raw": DataFrame 该人的映射行,
              "is_dominant": [True, False, ...] 与 meals 一一对应,
          }, ...
        }
    """
    meal = load_meal_table(raw_dir)
    zipdf = load_zip_table(raw_dir)

    valid_ids = set(meal["id_n"]) & set(zipdf["id_n"])
    subjects = {}

    for sid in sorted(valid_ids):
        rows = meal[meal["id_n"] == sid]
        zrows = zipdf[zipdf["id_n"] == sid]
        if dominant_only and not rows["is_dominant"].any():
            continue
        subjects[sid] = {
            "meals": list(zip(rows["beforeTime"], rows["afterTime"])),
            "meal_rows": rows,
            "is_dominant": rows["is_dominant"].tolist(),
            "zips": zrows["sensorData"].tolist(),
            "zip_raw": zrows,
        }
    return subjects


if __name__ == "__main__":
    # ===== 自测：跑通即模块可用 =====
    from paths import RAW

    print("=== norm_id 单测 ===")
    cases = [("HNU 21013", "HNU21013"), ("hnu21033", "HNU21033"),
             ("HNU21026j", "HNU21026J"), ("HNU21026", "HNU21026")]
    for raw, want in cases:
        got = norm_id(raw)
        print(f"  {raw!r:14s} → {got!r:12s} {'✓' if got == want else '✗ 期望 ' + want}")

    print("\n=== 名册统计 ===")
    subs = load_subjects(RAW)
    print("有效受试者数：", len(subs))
    n_meal = sum(len(v["meals"]) for v in subs.values())
    n_dom = sum(sum(v["is_dominant"]) for v in subs.values())
    n_zip = sum(len(v["zips"]) for v in subs.values())
    print(f"总顿数：{n_meal} | 惯用手顿：{n_dom} | 总 zip：{n_zip}")

    print("\n=== dominant_only 名册 ===")
    subs_d = load_subjects(RAW, dominant_only=True)
    print("有惯用手饭段的受试者数：", len(subs_d))
    print("样例：HNU21026 →", len(subs_d["HNU21026"]["meals"]), "顿,",
          sum(subs_d["HNU21026"]["is_dominant"]), "顿惯用手")

    print("\n=== 确认 HNU21026 与 HNU21026J 独立 ===")
    print("  HNU21026 在册：", "HNU21026" in subs, "| zip 数：", len(subs["HNU21026"]["zips"]))
    print("  HNU21026J 在册：", "HNU21026J" in subs, "| zip 数：", len(subs["HNU21026J"]["zips"]))
