"""Four-layer eating detection pipeline.

The pipeline keeps data preparation, feature extraction, model scoring, and
event decoding behind small interfaces so each layer can be tested or
replaced independently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .data_loader import SensorBundle, SensorStream
    from .metrics import compute_f1
except ImportError:
    from data_loader import SensorBundle, SensorStream
    from metrics import compute_f1


@dataclass(frozen=True)
class PipelineConfig:
    target_fs: float = 100.0
    window_s: float = 10.0
    step_s: float = 5.0
    enter_threshold: float = 0.60
    exit_threshold: float = 0.35
    enter_windows: int = 2
    exit_windows: int = 2
    smooth_s: float = 15.0
    min_duration_s: float = 60.0
    merge_gap_s: float = 30.0
    hour_range: tuple[float, float] | None = (5.0, 24.0)
    max_gap_s: float = 60.0


@dataclass(frozen=True)
class AlignedSegment:
    times: np.ndarray
    acc: np.ndarray
    gyro: np.ndarray
    ppg: np.ndarray


@dataclass(frozen=True)
class FeatureBatch:
    values: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    names: tuple[str, ...]


@dataclass(frozen=True)
class ProbabilitySequence:
    centers: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    probabilities: np.ndarray


def _stream_times(stream: SensorStream) -> np.ndarray:
    if stream.sample_times_ms is not None:
        return np.asarray(stream.sample_times_ms, dtype=float) / 1000.0
    return stream.start_ms / 1000.0 + np.arange(stream.n_samples) / stream.fs


def _interp_channels(stream: SensorStream | None, times: np.ndarray, channels: int) -> np.ndarray:
    if stream is None or stream.n_samples == 0:
        return np.zeros((times.size, channels), dtype=float)
    source_t = _stream_times(stream)
    source = np.asarray(stream.data, dtype=float)
    if source.ndim == 1:
        source = source[:, None]
    out = np.zeros((times.size, channels), dtype=float)
    for col in range(channels):
        values = source[:, min(col, source.shape[1] - 1)]
        finite = np.isfinite(values) & np.isfinite(source_t)
        if finite.sum() >= 2:
            out[:, col] = np.interp(times, source_t[finite], values[finite])
        elif finite.sum() == 1:
            out[:, col] = values[finite][0]
    return out


def _smooth_channels(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.shape[0] < 3:
        return values
    width = min(width | 1, values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1)
    if width < 3:
        return values
    pad = width // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(width, dtype=float) / width
    return np.column_stack([np.convolve(padded[:, i], kernel, mode="valid")
                            for i in range(values.shape[1])])


class DataLayer:
    """Clean streams and align IMU/PPG on the ACC clock without crossing gaps."""

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config

    def prepare(self, bundle: SensorBundle) -> list[AlignedSegment]:
        if bundle.acc is None or bundle.gyro is None:
            raise ValueError("the data layer requires ACC and GYRO streams")
        acc_t = _stream_times(bundle.acc)
        segments: list[AlignedSegment] = []
        smooth_width = max(1, int(round(self.config.target_fs * 0.05)))
        for sl in bundle.acc.segments(self.config.max_gap_s):
            times = acc_t[sl]
            if times.size < 2:
                continue
            acc = np.asarray(bundle.acc.data[sl], dtype=float)
            gyro = _interp_channels(bundle.gyro, times, 3)
            ppg = _interp_channels(bundle.ppg, times, 20)
            acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
            gyro = np.nan_to_num(gyro, nan=0.0, posinf=0.0, neginf=0.0)
            ppg = np.nan_to_num(ppg, nan=0.0, posinf=0.0, neginf=0.0)
            segments.append(AlignedSegment(times, _smooth_channels(acc, smooth_width),
                                           _smooth_channels(gyro, smooth_width),
                                           _smooth_channels(ppg, smooth_width)))
        return segments


def _stat_names(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{s}" for s in ("mean", "std", "rms", "range", "median_abs", "p10", "p90"))


FEATURE_NAMES = tuple(
    name
    for prefix in ("acc_x", "acc_y", "acc_z", "acc_mag", "gyro_x", "gyro_y", "gyro_z", "gyro_mag")
    for name in _stat_names(prefix)
) + (
    "ppg_mean", "ppg_std", "ppg_periodicity", "motion_std", "jerk_rms",
    "still_fraction", "low_band_energy", "mid_band_energy", "high_band_energy",
)


def _stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    return [float(np.mean(values)), float(np.std(values)),
            float(np.sqrt(np.mean(values ** 2))), float(np.ptp(values)),
            float(np.median(np.abs(values - median))),
            float(np.percentile(values, 10)), float(np.percentile(values, 90))]


def _band_energy(values: np.ndarray, fs: float) -> tuple[float, float, float]:
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(values.size, 1.0 / fs)
    total = float(np.sum(spectrum)) + 1e-9
    return tuple(float(np.sum(spectrum[(freqs >= lo) & (freqs < hi)]) / total)
                 for lo, hi in ((0.3, 3.0), (3.0, 8.0), (8.0, fs / 2 + 1)))


class FeatureLayer:
    """Create time, frequency, and activity-context features per fixed window."""

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config

    def transform(self, segments: list[AlignedSegment]) -> FeatureBatch:
        fs = self.config.target_fs
        width, step = int(round(fs * self.config.window_s)), int(round(fs * self.config.step_s))
        rows, starts, ends = [], [], []
        for segment in segments:
            n = min(len(segment.times), len(segment.acc), len(segment.gyro), len(segment.ppg))
            if n < width:
                continue
            acc, gyro, ppg, times = segment.acc[:n], segment.gyro[:n], segment.ppg[:n], segment.times[:n]
            for start in range(0, n - width + 1, step):
                stop = start + width
                row: list[float] = []
                for xyz in (acc[start:stop], gyro[start:stop]):
                    for col in (xyz[:, 0], xyz[:, 1], xyz[:, 2], np.linalg.norm(xyz, axis=1)):
                        row.extend(_stats(col))
                ppg_window = ppg[start:stop]
                ppg_signal = ppg_window[:, int(np.argmax(np.std(ppg_window, axis=0)))]
                motion = np.linalg.norm(acc[start:stop], axis=1)
                jerk = np.diff(acc[start:stop], axis=0)
                row.extend([float(np.mean(ppg_signal)), float(np.std(ppg_signal)),
                            float(_band_energy(ppg_signal, fs)[0])])
                row.extend([float(np.std(motion)),
                            float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1)))) if jerk.size else 0.0,
                            float(np.mean(np.abs(motion - np.median(motion)) < max(np.std(motion), 1e-9) * 0.25))])
                row.extend(_band_energy(motion, fs))
                rows.append(row)
                starts.append(float(times[start]))
                ends.append(float(times[stop - 1]))
        return FeatureBatch(np.asarray(rows, dtype=float).reshape(-1, len(FEATURE_NAMES)),
                            np.asarray(starts), np.asarray(ends), FEATURE_NAMES)


def _hour_ok(epoch_s: float, hour_range: tuple[float, float] | None) -> bool:
    if hour_range is None:
        return True
    hour = ((epoch_s % 86400.0) / 3600.0 + 8.0) % 24.0
    low, high = hour_range
    return low <= hour < high if low <= high else hour >= low or hour < high


class ModelLayer:
    """Fit a classifier and fuse its probability with the meal-time prior."""

    def __init__(self, config: PipelineConfig = PipelineConfig(), random_state: int = 42):
        self.config, self.random_state = config, random_state
        self.estimator = None

    def fit(self, batches: list[tuple[FeatureBatch, np.ndarray]]):
        from sklearn.ensemble import RandomForestClassifier
        features = np.concatenate([b.values for b, _ in batches], axis=0)
        labels = np.concatenate([np.asarray(y, dtype=np.uint8) for _, y in batches])
        if len(np.unique(labels)) != 2:
            raise ValueError("training data must contain both meal and non-meal windows")
        self.estimator = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced_subsample", n_jobs=-1,
                                                random_state=self.random_state)
        self.estimator.fit(features, labels)
        return self

    def predict(self, batch: FeatureBatch) -> ProbabilitySequence:
        if self.estimator is None:
            raise ValueError("model layer is not fitted")
        column = list(self.estimator.classes_).index(1)
        probabilities = self.estimator.predict_proba(batch.values)[:, column]
        centers = (batch.starts + batch.ends) / 2.0
        allowed = np.asarray([_hour_ok(t, self.config.hour_range) for t in centers])
        return ProbabilitySequence(centers, batch.starts, batch.ends,
                                   np.where(allowed, probabilities, 0.0))


class DecisionLayer:
    """Smooth probabilities and decode them with a hysteresis state machine."""

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config

    def decode(self, sequence: ProbabilitySequence) -> list[tuple[float, float]]:
        if sequence.probabilities.size == 0:
            return []
        width = max(1, int(round(self.config.smooth_s / self.config.step_s)))
        if width > 1:
            kernel = np.ones(width, dtype=float) / width
            padded = np.pad(sequence.probabilities, (width // 2, width - width // 2 - 1), mode="edge")
            smooth = np.convolve(padded, kernel, mode="valid")
        else:
            smooth = sequence.probabilities
        events, active, candidate = [], False, 0
        start = last_positive = None
        for i, probability in enumerate(smooth):
            if not active:
                candidate = candidate + 1 if probability >= self.config.enter_threshold else 0
                if candidate >= self.config.enter_windows:
                    active = True
                    start = float(sequence.starts[max(0, i - self.config.enter_windows + 1)])
                    last_positive = float(sequence.ends[i])
            elif probability >= self.config.exit_threshold:
                candidate = 0
                last_positive = float(sequence.ends[i])
            else:
                candidate += 1
                if candidate >= self.config.exit_windows:
                    events.append((float(start), float(last_positive)))
                    active, candidate, start, last_positive = False, 0, None, None
        if active and start is not None and last_positive is not None:
            events.append((float(start), float(last_positive)))
        merged = []
        for event_start, event_end in events:
            if merged and event_start - merged[-1][1] <= self.config.merge_gap_s:
                merged[-1] = (merged[-1][0], event_end)
            elif event_end - event_start >= self.config.min_duration_s:
                merged.append((event_start, event_end))
        return merged


class EatingPipeline:
    """End-to-end four-layer pipeline for training, inference, and evaluation."""

    def __init__(self, config: PipelineConfig = PipelineConfig(), random_state: int = 42):
        self.config = config
        self.data = DataLayer(config)
        self.features = FeatureLayer(config)
        self.model = ModelLayer(config, random_state=random_state)
        self.decision = DecisionLayer(config)

    def make_features(self, bundle: SensorBundle) -> FeatureBatch:
        return self.features.transform(self.data.prepare(bundle))

    def fit(self, training: list[tuple[SensorBundle, list[tuple[float, float]]]]):
        batches = []
        for bundle, truth in training:
            batch = self.make_features(bundle)
            labels = np.zeros(len(batch.starts), dtype=np.uint8)
            duration = np.maximum(batch.ends - batch.starts, 1e-9)
            for start, end in truth:
                overlap = np.maximum(0.0, np.minimum(batch.ends, end) - np.maximum(batch.starts, start))
                labels[overlap / duration >= 0.5] = 1
            batches.append((batch, labels))
        self.model.fit(batches)
        return self

    def predict(self, bundle: SensorBundle) -> list[tuple[float, float]]:
        return self.decision.decode(self.model.predict(self.make_features(bundle)))

    def evaluate(self, bundle: SensorBundle, truth: list[tuple[float, float]]) -> dict:
        predicted = self.predict(bundle)
        report = compute_f1(predicted, truth, include_matches=True)
        matches = report.get("matches", [])
        if matches:
            errors = []
            for match in matches:
                ps, pe = predicted[match.pred_index]
                ts, te = truth[match.true_index]
                errors.extend((abs(ps - ts), abs(pe - te)))
            report["boundary_mae_s"] = float(np.mean(errors))
        else:
            report["boundary_mae_s"] = None
        report["predicted_events"] = predicted
        return report

