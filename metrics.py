"""事件级评估：IoU 匹配、灵敏度、PPV 和 F1。

Rules used by the competition:

* an event is a time interval ``[start, end]``;
* one prediction is correct if its IoU with a ground-truth event is strictly
  greater than the threshold (default 0.25);
* matching is one-to-one (each event consumed at most once), solved by
  maximum bipartite matching;
* sensitivity = TP / (TP + FN), PPV = TP / (TP + FP), F1 = harmonic mean.
"""
from __future__ import annotations

from dataclasses import dataclass


def boundary_expansion_iou(length, delta):
    """IoU when boundary uncertainty expands the union by total width delta."""
    length, delta = float(length), float(delta)
    if length <= 0 or delta < 0:
        raise ValueError("length must be positive and delta must be non-negative")
    return length / (length + delta)


def boundary_delta_from_iou(length, score):
    """Invert IoU=L/(L+delta) to express total boundary expansion."""
    length, score = float(length), float(score)
    if length <= 0 or not 0 < score <= 1:
        raise ValueError("length must be positive and IoU must be in (0, 1]")
    return length * (1.0 / score - 1.0)


@dataclass(frozen=True)
class Event:
    """One meal interval; label is only for traceability."""

    start: float
    end: float
    label: object = None

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"event end({self.end}) < start({self.start})")


@dataclass(frozen=True)
class Match:
    """A matched pair; pred/true are indices into the input lists."""

    pred_index: int
    true_index: int
    iou: float


def _evt(x):
    """Normalise an Event / tuple / object with start-end to (start, end)."""
    if hasattr(x, "start"):
        s, e = x.start, x.end
    else:
        s, e = x[0], x[1]
    s, e = float(s), float(e)
    if e < s:
        raise ValueError(f"event end({e}) < start({s})")
    return s, e


def iou(pred_event, true_event):
    ps, pe = _evt(pred_event)
    ts, te = _evt(true_event)
    ov = min(pe, te) - max(ps, ts)
    if ov <= 0:
        return 0.0
    return ov / ((pe - ps) + (te - ts) - ov)


def match_events(pred_events, true_events, iou_threshold=0.25):
    """One-to-one greedy+augmenting maximum matching above IoU threshold."""
    pred = [_evt(x) for x in pred_events]
    true = [_evt(x) for x in true_events]
    n_pred, n_true = len(pred), len(true)

    cand = [[] for _ in range(n_pred)]
    for i in range(n_pred):
        for j in range(n_true):
            s = iou(pred[i], true[j])
            if s > iou_threshold:
                cand[i].append((j, s))
        cand[i].sort(key=lambda x: x[1], reverse=True)

    owner = [-1] * n_true

    def augment(p, used):
        if used[p]:
            return False
        used[p] = True
        for j, _ in cand[p]:
            if owner[j] < 0 or augment(owner[j], used):
                owner[j] = p
                return True
        return False

    for p in sorted(range(n_pred), key=lambda i: len(cand[i])):
        augment(p, [False] * n_pred)

    result = []
    for j, p in enumerate(owner):
        if p >= 0:
            score = next(s for (jj, s) in cand[p] if jj == j)
            result.append(Match(p, j, score))
    return result


def compute_f1(pred_events, true_events, iou_threshold=0.25, include_matches=False):
    """Full report including threshold/tp/fp/fn/sensitivity/ppv/f1."""
    pred_events = list(pred_events)
    true_events = list(true_events)
    ms = match_events(pred_events, true_events, iou_threshold)

    tp, fp, fn = len(ms), len(pred_events) - len(ms), len(true_events) - len(ms)
    sensitivity = tp / (tp + fn) if tp + fn else 1.0   # 召回率
    ppv = tp / (tp + fp) if tp + fp else 1.0           # 精确率
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if sensitivity + ppv else 0.0

    res = dict(threshold=float(iou_threshold), tp=tp, fp=fp, fn=fn,
               sensitivity=sensitivity, ppv=ppv, f1=f1,
               recall=sensitivity, precision=ppv)
    if include_matches:
        res["matches"] = ms
    return res
