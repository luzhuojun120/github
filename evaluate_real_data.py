"""Evaluate non-dominant-hand detection against meal annotations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

if __package__:
    from .data_loader import read_sensor_zip
    from .detect_nondominant import detect_nondominant_events, detect_nondominant_stitched
    from .metrics import compute_f1
    from .random_forest import load_model
else:
    from data_loader import read_sensor_zip
    from detect_nondominant import detect_nondominant_events, detect_nondominant_stitched
    from metrics import compute_f1
    from random_forest import load_model


def _hand_side(value):
    value = str(value or "")
    if "左" in value:
        return "left"
    if "右" in value:
        return "right"
    return None


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _plot_comparison(truths, predictions, matches, path, boundary_mae_s):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to write the event comparison plot") from exc
    origin = min([e["start"] for e in truths + predictions], default=0.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(3.0, 0.32 * (len(truths) + len(predictions)) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    for i, event in enumerate(truths):
        ax.barh(i, (event["end"] - event["start"]) / 60,
                left=(event["start"] - origin) / 60,
                height=0.65, color="#2878b5", label="Truth" if i == 0 else None)
    offset = len(truths)
    for i, event in enumerate(predictions):
        ax.barh(offset + i, (event["end"] - event["start"]) / 60,
                left=(event["start"] - origin) / 60,
                height=0.65, color="#ed8b2c", label="Prediction" if i == 0 else None)
    truth_y = {id(event): i for i, event in enumerate(truths)}
    pred_y = {id(event): offset + i for i, event in enumerate(predictions)}
    for match in matches:
        truth = match["truth"]
        pred = match["prediction"]
        ax.plot([(truth["start"] - origin) / 60, (pred["start"] - origin) / 60],
                [truth_y[id(truth)], pred_y[id(pred)]], color="#777777", alpha=0.45, linewidth=0.8)
    ax.set_yticks(range(len(truths) + len(predictions)))
    ax.set_yticklabels([f"Truth {i + 1}" for i in range(len(truths))] +
                       [f"Prediction {i + 1}" for i in range(len(predictions))])
    ax.set_xlabel("Minutes from first event")
    ax.set_title(f"Meal event comparison (matched boundary MAE: {boundary_mae_s:.1f} s)")
    ax.grid(axis="x", alpha=0.2)
    if truths or predictions:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(zip_dir, meals_csv, participant=None, mapping_csv=None,
             iou_threshold=0.25, min_duration_s=300.0,
             hour_range=(5.0, 24.0),
             model_path=None, plot_path=None, stitch_files=False):
    if not model_path:
        raise ValueError("--model is required; train a Random Forest before evaluation")
    model = load_model(model_path)
    zip_paths = sorted(Path(zip_dir).rglob("*.zip"))
    if not zip_paths:
        raise ValueError(f"no ZIP files found under {zip_dir}")
    by_name = {p.name: p for p in zip_paths}

    mapping_rows = _read_csv(mapping_csv) if mapping_csv else []
    mapping_by_name = {Path(row["sensorData"]).name: row for row in mapping_rows
                       if row.get("sensorData")}
    selected_paths = []
    available_intervals = []
    for name, path in by_name.items():
        row = mapping_by_name.get(name)
        if mapping_csv and not row:
            continue
        if row and participant and row.get("externalid") != participant:
            continue
        selected_paths.append(path)
        if row:
            available_intervals.append((int(row["timeStamp.startTime"]) / 1000.0,
                                        int(row["timeStamp.endTime"]) / 1000.0))

    meal_rows = _read_csv(meals_csv)
    truths = []
    for i, row in enumerate(meal_rows):
        if participant and row.get("externalid") != participant:
            continue
        dominant = _hand_side(row.get("dietaryHand"))
        worn = _hand_side(row.get("wearHand"))
        if not dominant or not worn or dominant == worn:
            continue
        start, end = int(row["beforeTime"]) / 1000.0, int(row["afterTime"]) / 1000.0
        if available_intervals and not any(a < end and b > start for a, b in available_intervals):
            continue
        truths.append({"index": i, "participant": row.get("externalid"),
                       "start": start, "end": end,
                       "dietary_type": row.get("dietaryType", "")})

    predictions = []
    errors = []
    if stitch_files:
        stitched = detect_nondominant_stitched(
            selected_paths, min_duration_s=min_duration_s,
            hour_range=hour_range, model=model)
        errors.extend(stitched.get("errors", []))
        predictions.extend({"file": "stitched", "start": float(s), "end": float(e)}
                           for s, e in stitched["events"])
    else:
        for path in selected_paths:
            try:
                bundle = read_sensor_zip(path)
                events = detect_nondominant_events(
                    bundle, min_duration_s=min_duration_s,
                    hour_range=hour_range, model=model)
                predictions.extend({"file": path.name, "start": float(s), "end": float(e)}
                                   for s, e in events)
            except Exception as exc:
                errors.append({"file": path.name, "error": repr(exc)})

    report = compute_f1([(p["start"], p["end"]) for p in predictions],
                        [(t["start"], t["end"]) for t in truths],
                        iou_threshold=iou_threshold, include_matches=True)
    matches = []
    matched_pred = set()
    matched_truth = set()
    for match in report.pop("matches"):
        matched_pred.add(match.pred_index)
        matched_truth.add(match.true_index)
        matches.append({"prediction": predictions[match.pred_index],
                        "truth": truths[match.true_index], "iou": match.iou})
    report.update({
        "participant": participant,
        "zip_count": len(selected_paths),
        "error_count": len(errors),
        "truth_count": len(truths),
        "prediction_count": len(predictions),
        "matches": matches,
        "unmatched_predictions": [p for i, p in enumerate(predictions) if i not in matched_pred],
        "unmatched_truth": [t for i, t in enumerate(truths) if i not in matched_truth],
        "errors": errors,
    })
    boundary_errors = [abs(match["prediction"][edge] - match["truth"][edge])
                       for match in matches for edge in ("start", "end")]
    boundary_mae = sum(boundary_errors) / len(boundary_errors) if boundary_errors else None
    report["boundary_mae_s"] = boundary_mae
    if plot_path:
        _plot_comparison(truths, predictions, matches, plot_path,
                         float("nan") if boundary_mae is None else boundary_mae)
        report["comparison_plot"] = str(plot_path)
    return report


def main():
    ap = argparse.ArgumentParser(description="Evaluate meal detection on annotated real sensor data")
    ap.add_argument("--zip-dir", required=True, help="directory containing sensor ZIP files")
    ap.add_argument("--meals", required=True, help="meal annotation CSV")
    ap.add_argument("--mapping", help="optional sensor download mapping CSV")
    ap.add_argument("--participant", help="externalid, for example HNU21026")
    ap.add_argument("--output", help="write JSON report to this path")
    ap.add_argument("--model", required=True, help="trained Random Forest joblib model")
    ap.add_argument("--plot", help="write truth-versus-prediction comparison PNG")
    ap.add_argument("--iou-threshold", type=float, default=0.25)
    ap.add_argument("--min-duration", type=float, default=300.0)
    ap.add_argument("--hour-range", default="5,24", help="start,end in UTC+8; none disables prior")
    ap.add_argument("--stitch-files", action="store_true",
                    help="按真实时间拼接目录中的多个 ZIP 后再评估")
    args = ap.parse_args()
    hour_range = None if args.hour_range.lower() == "none" else tuple(
        float(v) for v in args.hour_range.split(",", 1))
    plot_path = args.plot or (str(Path(args.output).with_suffix(".png"))
                              if args.output else "meal_comparison.png")
    report = evaluate(args.zip_dir, args.meals, participant=args.participant,
                       mapping_csv=args.mapping, iou_threshold=args.iou_threshold,
                       min_duration_s=args.min_duration, hour_range=hour_range,
                       model_path=args.model, plot_path=plot_path,
                       stitch_files=args.stitch_files)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("participant", "zip_count", "error_count", "truth_count",
                      "prediction_count", "tp", "fp", "fn", "sensitivity",
                      "ppv", "f1", "boundary_mae_s")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
