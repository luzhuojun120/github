"""非惯用手（PPG + ACC）检测器的数据预处理。

The original file only cut fixed-length windows and normalised them.  Real data
require extra steps:

* acceleration is raw ADC around gravity; motion features should be computed
  from a gravity-removed magnitude / jerk signal;
* repeated hand movements (eating gestures) are detected with an adaptive
  threshold against a local 60 s baseline instead of fixed units;
* long meal events are produced by smoothing 10 s decisions and merging short
  gaps, because people pause while chewing/conversing.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 窗口工具（保留原竞赛核心接口）
# ---------------------------------------------------------------------------


def sliding_window(signal, window_size, step, channels_first=False):
    """Cut a long signal into (n_win, window_size[, n_ch]) windows."""
    x = np.asarray(signal, dtype=float)
    if channels_first and x.ndim == 2:
        x = x.T  # (ch, n) -> (n, ch); time first
    w, s = int(window_size), int(step)
    if w <= 0 or s <= 0:
        raise ValueError("window_size / step must be positive integers")
    n = x.shape[0]
    if n < w:
        return np.empty((0, w) + x.shape[1:], dtype=float)
    idx = range(0, n - w + 1, s)
    return np.stack([x[i:i + w] for i in idx])


def normalize(windows, method="minmax"):
    """Per-window normalisation: minmax -> [0,1], zscore -> mean 0 / std 1."""
    x = np.asarray(windows, dtype=float)
    if x.ndim == 1:
        x = x[None, :]

    if method == "minmax":
        lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
        span = hi - lo
        out = np.zeros_like(x)
        np.divide(x - lo, span, out=out, where=span > 0)
        return out
    if method == "zscore":
        mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
        out = np.zeros_like(x)
        np.divide(x - mu, sd, out=out, where=sd > 0)
        return out
    raise ValueError("method only supports minmax or zscore")


def make_labels(windows, eat_events, step=None, min_overlap=0.5):
    """Label every window by overlap with one or more [start,end) events."""
    windows = np.asarray(windows, dtype=float)
    n, w = windows.shape[0], windows.shape[1]
    if n == 0:
        return np.zeros(0, dtype=int)
    if step is None:
        step = w
    starts = np.arange(n) * step
    ends = starts + w
    labels = np.zeros(n, dtype=int)

    events = np.asarray(eat_events, dtype=float)
    if events.size == 0:
        return labels
    events = events.reshape(-1, 2)
    for s, e in events:
        ov = np.minimum(ends, e) - np.maximum(starts, s)
        hit = np.maximum(ov, 0) >= min_overlap * w
        labels[hit] = 1
    return labels


# ---------------------------------------------------------------------------
# 真实数据检测器使用的信号工具
# ---------------------------------------------------------------------------


def moving_average(x, fs, window_s):
    """Moving-average with edge padding (window_s in seconds)."""
    x = np.asarray(x, dtype=float)
    k = int(round(fs * window_s))
    k = max(3, k | 1)
    if x.size < k:
        return np.full_like(x, np.nan if x.size == 0 else np.mean(x))
    p = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(p, np.ones(k) / k, mode="valid")


def detrend_ma(x, fs, window_s=2.0):
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return x - np.mean(x)
    return x - moving_average(x, fs, window_s)


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def mag_std_windows(acc, fs, window_s=10.0, step_s=5.0):
    """Per-window standard deviation of ||acc|| (a scale-free motion level)."""
    data = np.asarray(acc, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    mag = np.linalg.norm(data, axis=1)
    n = data.shape[0]
    w, s = int(round(fs * window_s)), int(round(fs * step_s))
    centers, stds = [], []
    for start in range(0, n - w + 1, s):
        seg = mag[start:start + w]
        if seg.size < max(8, w // 2):
            continue
        centers.append((start + w / 2) / fs)
        stds.append(float(np.std(seg)))
    return np.asarray(centers), np.asarray(stds)


def adaptive_burst_counts(acc, fs, window_s=10.0, step_s=5.0, baseline_s=60.0,
                          z_threshold=5.0):
    """Count adaptive motion bursts in each window.

    A burst is a positive crossing of ``jerk > median + z_threshold * MAD``,
    where median/MAD come from the previous minute.  Eating on the real data
    shows *more* such medium-magnitude bursts than walking/typing even though
    overall |acc| stays low.
    """
    data = np.asarray(acc, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    jerk = np.linalg.norm(np.diff(data, axis=0), axis=1)
    n = data.shape[0]
    w, s = int(round(fs * window_s)), int(round(fs * step_s))
    centers, counts = [], []
    end_win = w
    win_idx = 0
    while end_win <= n:
        center = (win_idx * s + w / 2) / fs
        i0, i1 = win_idx * s, end_win - 1  # jerk index aligned to acc diff
        jw = jerk[max(i0, 0):max(i1, 0)]
        # baseline: [-baseline_s-5, -5] seconds before the end of this window
        bl0 = int((win_idx * s + w - baseline_s * fs))
        bl1 = int((win_idx * s + w - 5 * fs))
        base = jerk[max(0, bl0):max(bl1, 0)]
        if base.size < int(fs * 25):
            base = jerk[max(0, int((win_idx * s + w - 30 * fs))):max(0, int(win_idx * s + w - 5 * fs))]
        med = float(np.median(base)) if base.size else float(np.median(jw)) if jw.size else 0.0
        mad = 1.4826 * float(np.median(np.abs(base - med))) + 1e-9 if base.size else 1e-9
        z = (jw - med) / max(mad, 1e-9) if jw.size else np.empty(0)
        over = z > z_threshold
        count = int(np.count_nonzero(over[1:] & ~over[:-1])) if over.size > 1 else 0
        centers.append(center)
        counts.append(count)
        win_idx += 1
        end_win += s
    return np.asarray(centers), np.asarray(counts)


def events_from_binary(times, positive, step_s, fraction=0.35, block_s=240.0,
                       gap_s=180.0, min_duration_s=300.0):
    """Turn sparse 10 s labels into meal events.

    A sliding block (default 4 min) must contain at least ``fraction`` positive
    windows; positive blocks are merged when separated by <= gap_s, and events
    shorter than min_duration_s are discarded.
    """
    times = np.asarray(times, float)
    positive = np.asarray(positive, bool)
    if positive.size == 0 or times.size == 0:
        return []
    if step_s <= 0:
        step_s = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    half = max(1, int(round(block_s / step_s / 2)))
    smooth = np.zeros(positive.size, dtype=float)
    for i in range(positive.size):
        lo, hi = max(0, i - half), min(positive.size, i + half + 1)
        seg = positive[lo:hi]
        smooth[i] = seg.sum() / seg.size if seg.size else 0.0
    state = smooth >= fraction

    # 合并间隔后的连续区段
    events = []
    start = None
    for i, (t, on) in enumerate(zip(times, state)):
        if on and start is None:
            start = t
        elif not on and start is not None:
            events.append([start, t])
            start = None
    if start is not None:
        events.append([start, times[-1]])

    merged = []
    for e in events:
        if merged and e[0] - merged[-1][1] <= gap_s:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    return [(s, e) for s, e in merged if (e - s) >= min_duration_s]


def plausible_meal_hour(epoch_s, hour_range=(5.0, 24.0)):
    """Meal-time prior from the real labels (all 288 meals are 08:00-23:xx
    except one 00:xx snack).  Used only to suppress obvious sleep periods."""
    if hour_range is None:
        return True
    low, high = hour_range
    hour = (epoch_s % 86400.0) / 3600.0 + 8.0  # data collected in UTC+8
    hour %= 24.0
    if low <= high:
        return low <= hour < high
    return hour >= low or hour < high
