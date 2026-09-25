"""Train a six-axis Random Forest from mapped sensor ZIPs and meal annotations."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from .data_loader import read_sensor_zip
    from .random_forest import label_windows, save_model, six_axis_windows, train_random_forest
else:
    from data_loader import read_sensor_zip
    from random_forest import label_windows, save_model, six_axis_windows, train_random_forest


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _side(value):
    value = str(value or "")
    if "左" in value:
        return "left"
    if "右" in value:
        return "right"
    return None


def train(zip_dir, meals_csv, mapping_csv, output, random_state=42):
    mapping = {Path(row["sensorData"]).name: row for row in _read_csv(mapping_csv)
               if row.get("sensorData")}
    meals = defaultdict(list)
    for row in _read_csv(meals_csv):
        dominant, worn = _side(row.get("dietaryHand")), _side(row.get("wearHand"))
        if dominant and worn and dominant != worn:
            meals[row.get("externalid", "")].append(
                (int(row["beforeTime"]) / 1000.0, int(row["afterTime"]) / 1000.0))

    feature_sets, label_sets = [], []
    zip_paths = sorted(Path(zip_dir).rglob("*.zip"))
    if not zip_paths:
        raise ValueError(f"no ZIP files found under {zip_dir}")
    used = 0
    for path in zip_paths:
        row = mapping.get(path.name)
        if not row:
            continue
        participant = row.get("externalid", "")
        bundle = read_sensor_zip(path)
        features, starts, ends, _ = six_axis_windows(bundle)
        if not len(features):
            continue
        interval_start = min(starts)
        interval_end = max(ends)
        truth = [(s, e) for s, e in meals[participant]
                 if s < interval_end and e > interval_start]
        feature_sets.append(features)
        label_sets.append(label_windows(starts, ends, truth))
        used += 1

    if not feature_sets:
        raise ValueError("no mapped ZIPs with usable ACC/GYRO data were found")
    features = np.concatenate(feature_sets)
    labels = np.concatenate(label_sets)
    model = train_random_forest(features, labels, random_state=random_state)
    save_model(model, output)
    return {"zip_count": used, "window_count": len(labels),
            "positive_windows": int(labels.sum()), "negative_windows": int((labels == 0).sum()),
            "model": str(output)}


def main():
    parser = argparse.ArgumentParser(description="Train six-axis Random Forest meal detector")
    parser.add_argument("--zip-dir", required=True)
    parser.add_argument("--meals", required=True)
    parser.add_argument("--mapping", required=True, help="sensorData-to-participant CSV")
    parser.add_argument("--output", default="meal_random_forest.joblib")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    print(train(args.zip_dir, args.meals, args.mapping, args.output, args.random_state))


if __name__ == "__main__":
    main()
