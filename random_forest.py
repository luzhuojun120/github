"""Six-axis window features and Random Forest training/inference helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .preprocess import sliding_window
except ImportError:
    from preprocess import sliding_window

WINDOW_POINTS = 500
STEP_POINTS = 250
FEATURE_NAMES = tuple(
    f"{sensor}_{axis}_{stat}"
    for sensor in ("acc", "gyro")
    for axis in ("x", "y", "z", "magnitude")
    for stat in ("mean", "std", "rms", "range", "median_abs", "p10", "p90")
)


def _sample_times(stream):
    if stream.sample_times_ms is not None:
        return np.asarray(stream.sample_times_ms, dtype=float) / 1000.0
    return stream.start_ms / 1000.0 + np.arange(stream.n_samples) / stream.fs


def six_axis_windows(bundle, window_points=WINDOW_POINTS, step_points=STEP_POINTS,
                     max_gap_s=60.0):
    """Return feature matrix, window start/end epoch seconds, and ACC fs."""
    acc, gyro = bundle.acc, bundle.gyro
    if acc is None or gyro is None:
        raise ValueError("Random Forest detection requires both ACC and GYRO streams")
    if acc.data.ndim != 2 or acc.data.shape[1] != 3 or gyro.data.ndim != 2 or gyro.data.shape[1] != 3:
        raise ValueError("ACC and GYRO streams must each contain three axes")

    acc_times = _sample_times(acc)
    gyro_times = _sample_times(gyro)
    all_features, starts, ends = [], [], []
    for segment in acc.segments(max_gap_s):
        a = np.asarray(acc.data[segment], dtype=float)
        t = acc_times[segment]
        if len(a) < window_points:
            continue
        # Align gyro samples to the ACC clock before cutting paired windows.
        valid = (gyro_times >= t[0]) & (gyro_times <= t[-1])
        if np.count_nonzero(valid) < 2:
            continue
        gt, gd = gyro_times[valid], np.asarray(gyro.data[valid], dtype=float)
        g = np.column_stack([np.interp(t, gt, gd[:, axis]) for axis in range(3)])
        paired = np.column_stack((a, g))
        windows = sliding_window(paired, window_points, step_points)
        if not len(windows):
            continue
        for idx, window in enumerate(windows):
            axes = [window[:, :3], window[:, 3:]]
            row = []
            for xyz in axes:
                values = np.column_stack((xyz, np.linalg.norm(xyz, axis=1)))
                for col in values.T:
                    median = np.median(col)
                    row.extend((np.mean(col), np.std(col), np.sqrt(np.mean(col ** 2)),
                                np.ptp(col), np.median(np.abs(col - median)),
                                np.percentile(col, 10), np.percentile(col, 90)))
            all_features.append(row)
            first = idx * step_points
            starts.append(float(t[first]))
            last = min(first + window_points - 1, len(t) - 1)
            ends.append(float(t[last]))
    return (np.asarray(all_features, dtype=float).reshape(-1, len(FEATURE_NAMES)),
            np.asarray(starts, dtype=float), np.asarray(ends, dtype=float), float(acc.fs))


def label_windows(starts, ends, events, min_overlap=0.5):
    starts, ends = np.asarray(starts), np.asarray(ends)
    labels = np.zeros(len(starts), dtype=np.uint8)
    for event_start, event_end in events:
        overlap = np.maximum(0.0, np.minimum(ends, event_end) - np.maximum(starts, event_start))
        duration = np.maximum(ends - starts, 1e-9)
        labels[overlap / duration >= min_overlap] = 1
    return labels


def train_random_forest(features, labels, random_state=42):
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to train the Random Forest model") from exc
    labels = np.asarray(labels, dtype=np.uint8)
    if len(np.unique(labels)) != 2:
        raise ValueError("training data must contain both meal and non-meal windows")
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=random_state)
    model.fit(np.asarray(features, dtype=float), labels)
    return {"estimator": model, "feature_names": FEATURE_NAMES,
            "window_points": WINDOW_POINTS, "step_points": STEP_POINTS,
            "threshold": 0.5}


def save_model(model, path):
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Install joblib to save the Random Forest model") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Install joblib to load the Random Forest model") from exc
    model = joblib.load(path)
    if tuple(model.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("model feature schema does not match this code version")
    return model
