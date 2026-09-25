# -*- coding: utf-8 -*-
"""
output.py —— 统一输出模块（适配层）
====================================
作用：所有分支的检测结果，统一通过本模块写成结果文件。

【⚠️ 为什么单独一个文件】
    结果文件的最终格式由官方《测试接口说明》规定，目前尚未拿到（2026-09-25 已在索取）。
    把输出逻辑集中在这里 → 官方要求到手后，**只改本文件**，特征/模型/评测全部不动。
    这就是"把会变的部分圈在一个小范围里"。

【当前采用的内部约定】（待官方确认，可能需要调整）
    事件格式:  [(start_ms, end_ms), ...]         毫秒时间戳
    结果文件:  CSV，每行一个事件
              subject_id,start_ms,end_ms,scenario
              HNU21026,1785650931000,1785652668000,dominant
    scenario: dominant（惯用手） / nondominant（非惯用手）

【用法】
    from output import make_rows, write_result
    rows = make_rows("HNU21026", segments, "dominant")
    write_result(rows, "result.csv")
"""

import csv
import os

# 结果文件的列（官方若要求改名，改这里）
HEADER = ["subject_id", "start_ms", "end_ms", "scenario"]

# 是否写表头（官方若要求无表头，改这里）
WRITE_HEADER = True


def make_rows(subject_id, segments, scenario):
    """
    把某个受试者的事件段转成结果行。

    参数
    ----
    subject_id : str
        受试者 ID，如 "HNU21026"
    segments : list of (int, int)
        事件段 [(start_ms, end_ms), ...]（各分支检测函数的返回值）
    scenario : str
        "dominant" 或 "nondominant"

    返回
    ----
    list of list，每行 [subject_id, start_ms, end_ms, scenario]
    """
    return [[subject_id, int(s), int(e), scenario] for (s, e) in segments]


def write_result(rows, out_path):
    """
    把结果行写成 CSV 文件。

    参数
    ----
    rows : list of list
        make_rows() 的返回值（或多个受试者的合并结果）
    out_path : str
        输出路径，如 "result.csv"

    注意
    ----
    · 编码用 UTF-8（官方如有特殊要求，改 open 的 encoding 参数）
    · 输出目录不存在时会自动创建
    """
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if WRITE_HEADER:
            w.writerow(HEADER)
        w.writerows(rows)

    return out_path


if __name__ == "__main__":
    # ===== 自测：跑通即说明本模块可用 =====
    print("=== 自测：把两段事件写成结果文件 ===")

    # 模拟两个受试者的检测结果
    seg_a = [(1785650931000, 1785652668000)]                 # HNU21026 检出 1 段
    seg_b = [(1785651000000, 1785652500000),                 # HNU21027 检出 2 段
             (1785655000000, 1785655300000)]

    rows = []
    rows += make_rows("HNU21026", seg_a, "dominant")
    rows += make_rows("HNU21027", seg_b, "nondominant")

    out = write_result(rows, "_selftest_output.csv")
    print("已写出:", out)
    print()
    print("=== 文件内容 ===")
    print(open(out, encoding="utf-8").read())
    print("=== 反推校验 ===")
    print("结果行数 =", len(rows))
    print("  段数来源：seg_a %d 段 + seg_b %d 段 = %d" % (len(seg_a), len(seg_b), len(seg_a) + len(seg_b)))
    print("  行数应等于段数：", len(rows) == len(seg_a) + len(seg_b))

    # 清理自测产物（自测只为验证逻辑，不留文件）
    os.remove(out)
    print()
    print("（自测产物已清理）")
