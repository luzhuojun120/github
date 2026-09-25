import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(1,os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_sensor_zip
from preprocess import make_labels,sliding_window
from features import extract_features,ACC_FS
from subjects import load_subjects
from paths import RAW, CACHE
IMU_AXES = ["ACC_X", "ACC_Z", "ACC_Y", "GYRO_X", "GYRO_Y", "GYRO_Z"]
WINDOW,STEP=500,250
WINDOW_MS=(WINDOW/ACC_FS)*1000

def load_subject_raw( info, raw_dir):
    dfs=[]
    for f in info["zips"]:
        p=os.path.join(raw_dir,"sensorData",os.path.basename(f))
        try:
            dfs.append(load_sensor_zip(p))
        except Exception as e:
            print(f"跳过损坏文件{os.path.basename(f)[:28]}...({type(e).__name__})")
    if not dfs:
        raise RuntimeError("该受试者无任何可读zip")
    big=pd.concat(dfs,ignore_index=True)
    big=big.sort_values("ACC_TIME").reset_index(drop=True)
    return big
def _split_segments(df,gap_ms=1000):
    t = df["ACC_TIME"].to_numpy()
    breaks = np.where(np.diff(t) > gap_ms)[0] + 1
    bound = np.concatenate([[0], breaks, [len(df)]])
    return [(bound[i], bound[i + 1]) for i in range(len(bound) - 1)]
def _subject_matrix(df):
    all_X,all_starts = [],[]
    for a,b in _split_segments(df):
        seg = df.iloc[a:b]
        if len(seg)<WINDOW:
            continue
        feasts=[]
        for ax in IMU_AXES:
            windows = sliding_window(seg[ax].to_numpy(), WINDOW, STEP)
            feasts.append(extract_features(windows,ACC_FS))
        X_seg=np.hstack(feasts)
        starts_seg = sliding_window(seg["ACC_TIME"].to_numpy(), WINDOW, STEP)[:, 0]
        all_X.append(X_seg)
        all_starts.append(starts_seg)
    return np.vstack(all_X),np.concatenate(all_starts)
def _window_is_dominant(starts,info):
    MEAL_MAX_GAP_MS=8*3600*1000
    WINDOW_MS=(WINDOW/ACC_FS)*1000
    meals=info["meals"]
    flags=info["is_dominant"]
    out= []
    for s in starts:
        b = s + WINDOW_MS
        best_dist,best_dom=None,False

        for (c, d) ,dom in zip(meals,flags):
            if b < c:           dist=c-b
            elif d<s:           dist=s-d
            else:               dist=0
            if best_dist is None or dist<best_dist:
                best_dist,best_dom=dist,dom
        out.append(bool(best_dom and best_dist<=MEAL_MAX_GAP_MS))
    return np.array(out,dtype=bool)
def build_subject(raw_dir,cache_dir,sid,info):
    cache_path=os.path.join(cache_dir,f"{sid}.npz")
    if os.path.exists(cache_path):
        return len(np.load(cache_path)["X"])
    try:
        big=load_subject_raw(info,raw_dir)
    except Exception as e:
        print(f"{sid}读取失败，跳过：{e}")
        return 0
    X,starts = _subject_matrix(big)
    meal_labels=make_labels(starts,WINDOW_MS,info["meals"])
    is_dom=_window_is_dominant(starts, info)
    os.makedirs(cache_dir,exist_ok=True)
    np.savez_compressed(os.path.join(cache_dir,f"{sid}.npz"),X=X,starts=starts,is_dominant=is_dom,meal_labels=meal_labels)
    return len(X)
def build_all(raw_dir,cache_dir,limit=None):
    subs=load_subjects(raw_dir)
    ids=sorted(subs.keys())
    if limit:ids=ids[:limit]
    for i,sid in enumerate(ids,1):
        n=build_subject(raw_dir,cache_dir,sid,subs[sid])
        print(f'[{i}/{len(ids)}]{sid}到{n}窗口')
if __name__ == "__main__":
    subs = load_subjects(RAW)
    sid = "HNU21001"
    n = build_subject(RAW, CACHE, sid, subs[sid])
    print("窗口数:", n, "| 缓存已写:", os.path.join(CACHE, f"{sid}.npz"))
    build_all(RAW, CACHE)
