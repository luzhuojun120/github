"""命令行入口：对真实华为传感器压缩包执行事件检测。

用法：
    python run_detection.py --zip sensorData-xxx.zip --model meal_random_forest.joblib
    python -m competition_core.run_detection --zip sensorData-xxx.zip --model meal_random_forest.joblib
"""
from __future__ import annotations

import argparse

if __package__:
    from .detect_nondominant import detect_nondominant_files
else:
    from detect_nondominant import detect_nondominant_files


def main():
    ap = argparse.ArgumentParser(description="六轴随机森林非惯用手进食事件检测")
    ap.add_argument("--zip", required=True, nargs="+", help="一个或多个 ZIP 路径，也可传入含 ZIP 的目录")
    ap.add_argument("--model", required=True, help="trained Random Forest joblib model")
    ap.add_argument("--mode", choices=["nondominant"], default="nondominant",
                    help="仅支持非惯用手；保留此参数以兼容旧命令")
    ap.add_argument("--hour-range", default="5,24",
                    help="允许检测的时间段，格式为起始小时,结束小时（UTC+8）；none 表示不限")
    ap.add_argument("--min-duration", type=float, default=300.0,
                    help="事件最短时长（秒）")
    ap.add_argument("--threshold", type=float, help="positive probability cutoff (defaults to model value)")
    ap.add_argument("--stitch-files", action="store_true",
                    help="按传感器时间拼接多个 ZIP，再跨文件合并事件")
    args = ap.parse_args()

    if args.hour_range.lower() == "none":
        hour_range = None
    else:
        lo, hi = args.hour_range.split(",")
        hour_range = (float(lo), float(hi))

    for result in detect_nondominant_files(
            args.zip, stitch=args.stitch_files, min_duration_s=args.min_duration,
            hour_range=hour_range, model=args.model, threshold=args.threshold):
        label = result["file"]
        if args.stitch_files:
            label = "%s (%d files)" % (label, len(result.get("files", [])))
        print("zip=%s" % label)
        for s, e in result["events"]:
            print(f"{s - result['t0']:.3f},{e - result['t0']:.3f}")


if __name__ == "__main__":
    main()
