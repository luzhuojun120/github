"""进食检测竞赛传感器数据读取。

The original module only understood one-dimensional signals exported as
.npy/.npz/.csv/.txt.  The real Huawei Research exports are zip archives
containing one large tab-separated ``collect_data*.txt`` with this layout::

    ACC_TIME  PPG_TIME  GYRO_TIME
    PPG1 ... PPG20           # columns 21-44 are always 0 on these watches
    ACC_X ACC_Y ACC_Z
    GYRO_X GYRO_Y GYRO_Z

Measured from the real data set:

* ACC/GYRO rows are interleaved and one row order corresponds to roughly
  100 Hz (some rows share the same millisecond timestamp because the
  recorder writes one packet per line).
* PPG rows carry ~15 samples per ~600 ms packet, i.e. about 25 Hz.
* Only PPG1..PPG20 contain data; PPG21..PPG44 are unused.
* Raw ACC values are unsigned ADC counts (not yet scaled to m/s^2).

This module keeps the old ``load_signal`` API and adds a zip loader that
returns aligned sensor streams.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PPG_FS = 25.0
DEFAULT_ACC_FS = 100.0
DEFAULT_GYRO_FS = 100.0

# 原始文件含 44 个 PPG 列，本数据集仅使用前 20 列。
PPG_CHANNELS = 20


@dataclass
class SensorStream:
    """A single uniformly sampled stream reconstructed from the raw export.

    ``data`` is one-dimensional for a single channel, or (n_samples, n_ch) for
    multi-channel data.  Times are stored as a start time plus a nominal fs
    because several raw rows share the same recorder timestamp.
    """

    name: str
    start_ms: float          # Unix epoch ms of the first raw packet
    fs: float                # nominal samples / second
    data: np.ndarray
    sample_times_ms: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs if self.fs else 0.0

    def time(self, epoch: bool = True) -> np.ndarray:
        """Sample times in seconds.  epoch=False gives time since first sample."""
        if self.sample_times_ms is not None:
            t = self.sample_times_ms / 1000.0
            return t if epoch else t - t[0]
        t0 = self.start_ms / 1000.0 if epoch else 0.0
        return t0 + np.arange(self.n_samples, dtype=float) / self.fs

    def segments(self, max_gap_s=60.0):
        """Return contiguous sample slices, splitting at recorder time gaps."""
        if self.sample_times_ms is None or self.n_samples < 2:
            return [slice(0, self.n_samples)] if self.n_samples else []
        cuts = np.flatnonzero(np.diff(self.sample_times_ms) > max_gap_s * 1000.0) + 1
        edges = np.r_[0, cuts, self.n_samples]
        return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    def indices_between(self, start_s: float, end_s: float, epoch: bool = True):
        t0 = self.start_ms / 1000.0 if epoch else 0.0
        first = max(0, int(np.floor((start_s - t0) * self.fs)))
        last = min(self.n_samples, int(np.ceil((end_s - t0) * self.fs)) + 1)
        return slice(max(first, 0), max(last, first))


@dataclass
class SensorBundle:
    """Parsed contents of one sensor zip."""

    meta: dict
    ppg: SensorStream | None = None
    acc: SensorStream | None = None
    gyro: SensorStream | None = None

    def stream(self, name: str) -> SensorStream | None:
        return getattr(self, name, None)

    @property
    def t0_s(self) -> float:
        """Earliest sample among streams, in epoch seconds."""
        starts = [s.start_ms for s in (self.ppg, self.acc, self.gyro) if s is not None]
        return (min(starts) if starts else 0.0) / 1000.0


def _pick_column(arr, column):
    x = arr.ravel() if arr.ndim == 1 else arr[:, column]
    if x.size == 0:
        raise ValueError("read signal is empty")
    return x


def load_signal(path, column=0):
    """Load a one-dimensional PPG/IMU signal from npy/npz/csv/txt (legacy API)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".npy":
        return _pick_column(np.load(p), column)
    if p.suffix.lower() == ".npz":
        with np.load(p) as f:
            return _pick_column(f[f.files[0]], column)

    rows = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [c for c in line.split(",") if c] if "," in line else line.split()
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            continue  # 表头或注释行
        rows.append(vals)
    if not rows:
        raise ValueError("file has no parseable numeric rows")
    return _pick_column(np.asarray(rows, dtype=float), column)


def _estimate_fs(timestamps_ms, nominal_fs, min_gap_ms):
    """Estimate effective rate from packet timestamps and rows per packet."""
    n = len(timestamps_ms)
    if n < 2:
        return float(nominal_fs)
    starts = []
    for t in timestamps_ms:
        if not starts or t != starts[-1]:
            starts.append(t)
    if len(starts) < 2:
        return float(nominal_fs)

    gaps = np.diff(starts)
    gaps = gaps[gaps > 0]
    typical_gap = float(np.median(gaps)) if gaps.size else float(min_gap_ms)
    # Estimate rate from the typical packet cadence. Long outages must not
    # dilute the nominal sample rate used to process the individual runs.
    span_s = len(starts) * typical_gap / 1000.0
    fs = n / max(span_s, 1e-6)
    # Allow some dropout but never trust a nonsense estimate.
    fs = float(np.clip(fs, nominal_fs * 0.5, nominal_fs * 1.6))
    return fs


def _make_stream(name, rows, nominal_fs, min_gap_ms):
    if not rows:
        return None
    ts = np.asarray([r[0] for r in rows], dtype=float)
    fs = _estimate_fs(ts, nominal_fs, min_gap_ms)
    cols = rows[0][1]
    if isinstance(cols, tuple):
        data = np.asarray([r[1] for r in rows], dtype=float)
    else:
        data = np.asarray([r[1] for r in rows], dtype=float).ravel()
    return SensorStream(name=name, start_ms=float(ts[0]), fs=fs, data=data,
                        sample_times_ms=ts)


def read_sensor_zip(path):
    """Parse a Huawei Research ``sensorData-*.zip`` and return a SensorBundle.

    Parameters
    ----------
    path : str or Path
        Path to a ``sensorData-<ts>-<uuid>.zip`` downloaded from the Huawei
        Research platform.

    Returns
    -------
    SensorBundle
        Contains ``ppg`` (n,20), ``acc`` (n,3) and ``gyro`` (n,3) streams plus
        metadata.  Streams whose sensor is absent or all-zero are ``None``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    with zipfile.ZipFile(p) as zf:
        txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not txt_names:
            raise ValueError(f"{p.name}: no collect_data*.txt inside zip")
        txt_name = txt_names[0]
        info_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
        meta = {"zip": p.name, "txt": txt_name}
        if info_names:
            meta["info_json"] = info_names[0]

        acc_rows, ppg_rows, gyro_rows = [], [], []
        with zf.open(txt_name) as raw:
            header_line = raw.readline()
            try:
                header = header_line.decode("utf-8-sig").rstrip("\r\n").split("\t")
            except UnicodeDecodeError:
                header = header_line.decode("gb18030", errors="ignore").rstrip("\r\n").split("\t")
            lower = [h.lower() for h in header]

            def idx(name):
                try:
                    return lower.index(name.lower())
                except ValueError as exc:
                    raise ValueError(f"{txt_name}: column {name!r} not in header {header[:8]}...") from exc

            i_act = idx("ACC_TIME")
            i_ppt = idx("PPG_TIME")
            i_gyt = idx("GYRO_TIME")
            i_ppg1 = idx("PPG1")
            i_acx = idx("ACC_X")
            i_gxx = idx("GYRO_X")

            for line in raw:
                parts = line.rstrip(b"\r\n").split(b"\t")
                if len(parts) <= max(i_ppt, i_act, i_gyt):
                    continue
                try:
                    act = float(parts[i_act]) if len(parts) > i_act else 0.0
                    ppt = float(parts[i_ppt]) if len(parts) > i_ppt else 0.0
                    gyt = float(parts[i_gyt]) if len(parts) > i_gyt else 0.0
                except ValueError:
                    continue

                if act > 0 and len(parts) > i_acx + 2:
                    try:
                        acc_rows.append((act, (float(parts[i_acx]), float(parts[i_acx + 1]), float(parts[i_acx + 2]))))
                    except ValueError:
                        pass
                if gyt > 0 and len(parts) > i_gxx + 2:
                    try:
                        gv = (float(parts[i_gxx]), float(parts[i_gxx + 1]), float(parts[i_gxx + 2]))
                        if any(gv):
                            gyro_rows.append((gyt, gv))
                    except ValueError:
                        pass
                if ppt > 0 and len(parts) > i_ppg1:
                    try:
                        vals = []
                        for k in range(PPG_CHANNELS):
                            if len(parts) <= i_ppg1 + k:
                                vals.append(0.0)
                                continue
                            s = parts[i_ppg1 + k].strip()
                            vals.append(float(s) if s not in (b"", b"0", b"0.0") else 0.0)
                        if any(vals):
                            ppg_rows.append((ppt, tuple(vals)))
                    except ValueError:
                        pass

    bundle = SensorBundle(meta=meta)
    bundle.ppg = _make_stream("ppg", ppg_rows, DEFAULT_PPG_FS, 600.0)
    bundle.acc = _make_stream("acc", acc_rows, DEFAULT_ACC_FS, 95.0)
    bundle.gyro = _make_stream("gyro", gyro_rows, DEFAULT_GYRO_FS, 95.0)
    return bundle


def load_data(path_or_zip, column=0):
    """Convenience dispatch: zip -> SensorBundle, other file -> np.ndarray."""
    p = Path(path_or_zip)
    if p.is_dir():
        raise ValueError("expected a file; pass a sensor zip or a signal file")
    if p.suffix.lower() == ".zip":
        return read_sensor_zip(p)
    return load_signal(p, column=column)
