"""合成鲁棒性测试：比较进食与喝水、说话、打字、走路等负样本。"""
import argparse

import numpy as np

if __package__:
    from .detect_nondominant import detect_nondominant
    from .metrics import compute_f1
else:
    from detect_nondominant import detect_nondominant
    from metrics import compute_f1


def make(kind, fs=25.0, seconds=12.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = 0.02 * rng.normal(size=n)
    if kind == "eat":
        x += 0.25 * np.sin(2 * np.pi * 1.2 * t) + 0.08 * np.sin(2 * np.pi * 2.4 * t)
    elif kind == "drink":
        x += 0.15 * np.sin(2 * np.pi * .7 * t) + 0.18 * np.sin(2 * np.pi * 3 * t)
    elif kind == "talk":
        x += 0.22 * np.sin(2 * np.pi * 5 * t) + 0.12 * rng.normal(size=n)
    elif kind == "type":
        x += 0.3 * (np.sin(2 * np.pi * 8 * t) > 0) - 0.15
    elif kind == "walk":
        x += 0.35 * np.sin(2 * np.pi * 1.8 * t)
    return x


def run(seed=0):
    kinds = ["eat", "drink", "talk", "type", "walk"]
    out = {}
    for k in kinds:
        vals = [detect_nondominant(make(k, seed=seed + i), return_score=True)[0] for i in range(20)]
        out[k] = {"positive_rate": float(np.mean(vals))}
    preds = [(0, 12)] if out["eat"]["positive_rate"] > 0.3 else []
    out["event_f1"] = compute_f1(preds, [(0, 12)])["f1"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    print(run(ap.parse_args().seed))
