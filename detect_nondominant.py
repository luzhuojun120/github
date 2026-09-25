"""非惯用手（佩戴 PPG）进食检测。

The original implementation assumed one clean PPG waveform (e.g. a 1-D
synthetic sine).  On the real Huawei watch data:

* PPG is 20 parallel raw channels at ~25 Hz; columns 3-8 carry the most
  periodic signal and are used as quality/HR evidence, while columns 1-2 and
  9-20 are noisier duplicate channels;
* session inference uses a trained Random Forest over aligned ACC and GYRO
  axes; the PPG window helpers remain for legacy callers.

Both APIs are provided:

* ``extract_ppg_features(window, fs)`` / ``detect_nondominant(window, fs)``:
  fixed-window feature/decision (legacy + quick tests).
* ``detect_nondominant_events(bundle, **kwargs)``: full session segmentation,
  returns [start_s, end_s] in Unix epoch seconds (use t0 to make relative).
* ``NondominantMealDetector``: streaming wrapper compatible with the old
  ``update(samples)`` loop.
"""
from __future__ import annotations

import os
import numpy as np

try:
    from .data_loader import SensorBundle
    from .preprocess import detrend_ma, events_from_binary, moving_average, plausible_meal_hour
except ImportError:  # 在目录中直接运行时使用无包名前缀导入
    from data_loader import SensorBundle
    from preprocess import detrend_ma, events_from_binary, moving_average, plausible_meal_hour

DEFAULT_FS = 25.0
MIN_PEAKS = 4


def _channel_quality(col, fs):
    """Return (dominant hr bpm, spectral ratio) for one PPG channel window."""
    x = np.asarray(col, dtype=float)
    if x.size < max(16, int(4 * fs)) or np.std(x) < 1e-6:
        return np.nan, 0.0
    y = detrend_ma(x, fs, 2.0)
    y = y - np.mean(y)
    n = y.size
    if n < 32:
        return np.nan, 0.0
    win = np.hanning(n)
    P = np.abs(np.fft.rfft(y * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= 0.5) & (freqs <= 3.5)
    if mask.sum() < 2:
        return np.nan, 0.0
    pm = P[mask]
    i = int(np.argmax(pm))
    dom = float(freqs[mask][i])
    ratio = float(pm[i] / (np.mean(pm) + 1e-12))
    if ratio < 2.0:
        return np.nan, 0.0
    return dom * 60.0, ratio


def _best_channel(window, fs):
    x = np.asarray(window, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    best = None
    best_score = 0.0
    for c in range(x.shape[1]):
        _, ratio = _channel_quality(x[:, c], fs)
        if ratio > best_score:
            best_score = ratio
            best = c
    return best, best_score


def extract_ppg_features(window, fs=DEFAULT_FS):
    """Compute PPG quality / heart-rate features from a 1-D or 20-channel window.

    Returns
    -------
    dict with keys used by the rest of the package:
        signal_quality  0..1 spectral periodicity of the best channel
        mean_hr         bpm, NaN when no periodic pulse is found
        sdnn/rmssd      NaN (25 Hz raw data are too coarse for beat-to-beat HRV)
        n_peaks / motion_artifact  kept for API compatibility
    """
    x = np.asarray(window, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    quality_floor = float(np.isfinite(x).mean())
    if x.size == 0:
        return dict(n_peaks=0, mean_hr=np.nan, sdnn=np.nan, rmssd=np.nan,
                    signal_quality=0.0, motion_artifact=1.0)

    ch, best_ratio = _best_channel(x, fs)
    out = dict(n_peaks=0, mean_hr=np.nan, sdnn=np.nan, rmssd=np.nan,
               signal_quality=0.0, motion_artifact=0.0, used_channel=ch)
    if ch is None:
        out["signal_quality"] = quality_floor * 0.1
        return out

    hr, ratio = _channel_quality(x[:, ch], fs)
    # On the real data a clean 10 s pulse window has ratio ~8-20; clipping keeps
    # the score in [0,1] and prevents a single perfect FFT from dominating.
    out["signal_quality"] = float(np.clip(ratio / 8.0, 0.0, 1.0))
    out["mean_hr"] = hr if np.isfinite(hr) else np.nan
    out["n_peaks"] = 1 if np.isfinite(hr) else 0
    out["motion_artifact"] = 0.0 if ratio >= 3.0 else float(np.clip((3.0 - ratio) / 3.0, 0.0, 1.0))
    return out


def detect_nondominant(window, fs=DEFAULT_FS, rest_hr=75.0, rmssd_max=45.0,
                       min_quality=0.5, imu=None, return_score=False):
    """Window-level decision used by tests / old streaming API.

    The heuristic rests on PPG periodicity quality: a window is "eating-like"
    only when a clean pulse exists.  The real session detector combines this
    with ACC/GYRO Random Forest features (see detect_nondominant_events).
    """
    f = extract_ppg_features(window, fs)
    score = float(f["signal_quality"])
    if imu is not None:
        a = np.asarray(imu, dtype=float)
        if a.ndim == 2 and a.shape[0] > 2:
            motion = float(np.std(np.linalg.norm(a, axis=1)))
            # Real ACC std during meals is roughly 60..300; penalise heavy activity.
            if motion > 450:
                score *= 0.5
    result = (score >= min_quality, score)
    return result if return_score else result[0]


def _hour_ok_for_windows(centers, hour_range):
    if hour_range is None:
        return np.ones(centers.size, dtype=bool)
    return np.asarray([plausible_meal_hour(t, hour_range) for t in centers], dtype=bool)


def detect_nondominant_events(bundle, motion_low=0.0, motion_high=300.0,
                              burst_min=0, fraction=0.35, block_s=240.0,
                              gap_s=180.0, min_duration_s=300.0,
                              window_s=10.0, step_s=5.0,
                              ppg_ratio_min=3.0, ppg_min_covered=0.25,
                              hour_range=(5.0, 24.0), max_gap_s=60.0,
                              model=None, threshold=None):
    """Detect eating events in a non-dominant-hand SensorBundle.

    A trained Random Forest scores 500-point six-axis windows at a 250-point
    step. ``hour_range`` can optionally remove obvious sleep windows. The
    motion-related legacy arguments remain accepted for caller compatibility.

    Returns
    -------
    list[(start_s, end_s)] in Unix epoch seconds.
    """
    if model is None:
        raise ValueError("a trained Random Forest is required; train one with train_model.py")
    if isinstance(model, (str, bytes, os.PathLike)):
        try:
            from .random_forest import load_model
        except ImportError:
            from random_forest import load_model
        model = load_model(model)
    if isinstance(bundle, dict):
        bundle = SensorBundle(meta=bundle.get("meta", {}), ppg=bundle.get("ppg"),
                              acc=bundle.get("acc"), gyro=bundle.get("gyro"))
    try:
        from .random_forest import six_axis_windows
    except ImportError:
        from random_forest import six_axis_windows
    features, starts, ends, fs = six_axis_windows(
        bundle, window_points=model.get("window_points", 500),
        step_points=model.get("step_points", 250), max_gap_s=max_gap_s)
    if not len(features):
        return []
    estimator = model["estimator"]
    positive_column = list(estimator.classes_).index(1)
    probabilities = estimator.predict_proba(features)[:, positive_column]
    limit = float(model.get("threshold", 0.5) if threshold is None else threshold)
    centers = (starts + ends) / 2.0
    positive = (probabilities >= limit) & _hour_ok_for_windows(centers, hour_range)
    step_seconds = 250.0 / fs
    boundaries = np.flatnonzero(np.diff(centers) > max_gap_s) + 1
    chunks = np.split(np.arange(len(centers)), boundaries)
    events = []
    for indices in chunks:
        if not len(indices):
            continue
        events.extend(events_from_binary(centers[indices], positive[indices], step_seconds,
                                         fraction=fraction, block_s=block_s,
                                         gap_s=gap_s, min_duration_s=min_duration_s))
    return sorted(events)


def _sensor_files(paths):
    """Expand ZIP paths/directories once, preserving a deterministic order."""
    from pathlib import Path

    files = []
    seen = set()
    for item in paths:
        path = Path(item)
        candidates = sorted(path.glob("*.zip")) if path.is_dir() else [path]
        for candidate in candidates:
            key = str(candidate.resolve())
            if candidate.suffix.lower() == ".zip" and key not in seen:
                seen.add(key)
                files.append(candidate)
    return files


def detect_nondominant_stitched(paths, **kwargs):
    """Detect a set of sensor ZIPs as one time-ordered session.

    Each archive is windowed independently, then model decisions are sorted by
    recorder timestamps. Event smoothing is applied across archive boundaries,
    while gaps larger than ``max_gap_s`` remain hard boundaries. This supports
    adjacent exports such as the 35-file HNU21026 set without fabricating
    samples between archives.
    """
    try:
        from .data_loader import read_sensor_zip
        from .random_forest import load_model, six_axis_windows
    except ImportError:
        from data_loader import read_sensor_zip
        from random_forest import load_model, six_axis_windows

    files = _sensor_files(paths)
    if not files:
        return {"file": "stitched", "files": [], "t0": 0.0, "events": [],
                "errors": [], "error_count": 0}
    model = kwargs.pop("model", None)
    if model is None:
        raise ValueError("a trained Random Forest is required; train one with train_model.py")
    if isinstance(model, (str, bytes, os.PathLike)):
        model = load_model(model)

    max_gap_s = float(kwargs.pop("max_gap_s", 60.0))
    hour_range = kwargs.pop("hour_range", (5.0, 24.0))
    threshold = kwargs.pop("threshold", None)
    fraction = float(kwargs.pop("fraction", 0.35))
    block_s = float(kwargs.pop("block_s", 240.0))
    gap_s = float(kwargs.pop("gap_s", 180.0))
    min_duration_s = float(kwargs.pop("min_duration_s", 300.0))
    window_points = int(model.get("window_points", 500))
    step_points = int(model.get("step_points", 250))
    estimator = model["estimator"]
    try:
        positive_column = list(estimator.classes_).index(1)
    except ValueError as exc:
        raise ValueError("model estimator does not contain positive class 1") from exc
    limit = float(model.get("threshold", 0.5) if threshold is None else threshold)

    chunks = []
    loaded = []
    errors = []
    for path in files:
        try:
            bundle = read_sensor_zip(path)
            features, starts, ends, fs = six_axis_windows(
                bundle, window_points=window_points, step_points=step_points,
                max_gap_s=max_gap_s)
            if len(features):
                probabilities = estimator.predict_proba(features)[:, positive_column]
                centers = (starts + ends) / 2.0
                positive = (probabilities >= limit) & _hour_ok_for_windows(centers, hour_range)
                chunks.append((centers, positive, float(fs)))
            loaded.append((path, bundle))
        except Exception as exc:
            errors.append({"file": path.name, "error": repr(exc)})

    if not chunks:
        t0 = min((b.t0_s for _, b in loaded), default=0.0)
        return {"file": "stitched", "files": [p.name for p in files], "t0": t0,
                "events": [], "errors": errors, "error_count": len(errors)}

    centers = np.concatenate([v[0] for v in chunks])
    positive = np.concatenate([v[1] for v in chunks])
    rates = np.asarray([v[2] for v in chunks], dtype=float)
    order = np.argsort(centers, kind="stable")
    centers, positive = centers[order], positive[order]
    diffs = np.diff(centers)
    positive_diffs = diffs[(diffs > 0) & (diffs <= max_gap_s)]
    default_step = (float(np.median(positive_diffs)) if positive_diffs.size
                    else float(step_points / np.median(rates)))
    boundaries = np.flatnonzero(diffs > max_gap_s) + 1
    events = []
    for indices in np.split(np.arange(len(centers)), boundaries):
        if not len(indices):
            continue
        local_diffs = np.diff(centers[indices])
        local_positive = local_diffs[local_diffs > 0]
        step_s = float(np.median(local_positive)) if local_positive.size else default_step
        events.extend(events_from_binary(
            centers[indices], positive[indices], step_s,
            fraction=fraction, block_s=block_s, gap_s=gap_s,
            min_duration_s=min_duration_s))
    t0 = min((b.acc.start_ms / 1000.0 if b.acc is not None else b.t0_s
              for _, b in loaded), default=float(centers[0]))
    return {"file": "stitched", "files": [p.name for p in files], "t0": t0,
            "events": sorted(events), "errors": errors, "error_count": len(errors)}


def detect_nondominant_files(paths, stitch=False, **kwargs):
    """Detect ZIPs independently, or stitch them by recorder time.

    ``stitch=False`` retains the original per-file result shape. Set
    ``stitch=True`` for adjacent exports that should form one session.
    """
    if stitch:
        return [detect_nondominant_stitched(paths, **kwargs)]
    try:
        from .data_loader import read_sensor_zip
    except ImportError:
        from data_loader import read_sensor_zip

    results = []
    for path in _sensor_files(paths):
        bundle = read_sensor_zip(path)
        bundle_t0 = bundle.acc.start_ms / 1000.0 if bundle.acc is not None else bundle.t0_s
        results.append({"file": path.name, "t0": bundle_t0,
                        "events": detect_nondominant_events(bundle, **kwargs)})
    return results


class NondominantMealDetector:
    """Streaming detector kept for the original CLI/test interface.

    Call ``update(chunk)`` repeatedly (each chunk is samples of one sensor
    stream) and optionally ``update(..., imu=acc_chunk)``; finish with
    ``finish()``.  It returns events with the same time convention as the old
    code (seconds from the start of the stream).
    """

    def __init__(self, fs=DEFAULT_FS, window_s=12.0, step_s=2.0,
                 start_prob=0.6, end_prob=0.35, start_windows=2,
                 end_gap_s=45.0, refractory_s=20.0, imu_fs=100.0):
        self.fs = float(fs)
        self.imu_fs = float(imu_fs)
        self.window = max(2, int(window_s * self.fs))
        self.step = max(1, int(step_s * self.fs))
        self.start_prob = start_prob
        self.end_gap = end_gap_s
        self.refractory = float(refractory_s)
        self._buf = np.empty(0)
        self._imu_buf = np.empty((0, 3))
        self.t = 0.0
        self.active = False
        self.candidate = []
        self.last_positive = None
        self.start = None
        self.events = []
        self._last_end = -np.inf

    def update(self, samples, imu=None):
        x = np.asarray(samples, dtype=float).ravel()
        self._buf = np.r_[self._buf, x]
        if imu is not None:
            im = np.asarray(imu, dtype=float)
            self._imu_buf = np.r_[self._imu_buf, im.reshape(-1, im.shape[-1])]
        out = []
        while self._buf.size >= self.window:
            w = self._buf[:self.window]
            iw = self._imu_buf[:self.window // max(1, int(self.fs / self.imu_fs))] \
                if self._imu_buf.shape[0] else None
            eat, prob = detect_nondominant(w, self.fs, imu=iw, return_score=True)
            s, e = self.t, self.t + self.window / self.fs
            if not self.active:
                self.candidate = (self.candidate + [(s, e, prob)])[-2:]
                if (len(self.candidate) == 2 and
                        all(v[2] >= self.start_prob for v in self.candidate) and
                        s - self._last_end >= self.refractory):
                    self.active = True
                    self.start = self.candidate[0][0]
                    self.last_positive = e
                    self.candidate = []
            elif eat and prob >= self.start_prob:
                self.last_positive = e
            elif self.last_positive is not None and e - self.last_positive >= self.end_gap:
                ev = (self.start, self.last_positive)
                self.events.append(ev)
                out.append(ev)
                self._last_end = self.last_positive
                self.active = False
                self.start = None
                self.last_positive = None
            self.t += self.step / self.fs
            self._buf = self._buf[self.step:]
            if self._imu_buf.shape[0]:
                self._imu_buf = self._imu_buf[int(self.step * self.imu_fs / self.fs):]
        return out

    def finish(self):
        if self.active and self.last_positive is not None:
            ev = (self.start, self.last_positive)
            self.events.append(ev)
            self.active = False
            return ev
        return None
