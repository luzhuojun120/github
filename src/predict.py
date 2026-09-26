import os, sys, gzip, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
import numpy as np
from scipy.ndimage import gaussian_filter1d
import glob
from paths import MODEL, BASE
from data_loader import load_sensor_zip
from preprocess import sliding_window
from features import extract_features
from segment_meals import segment_meals, merge_close
INPUT_DIR  = os.path.join(BASE, "test_data")
OUTPUT_CSV = os.path.join(BASE, "answer.csv")
def load_model():
    """读 model.pkl.gz → 返回 payload"""
    with gzip.open(MODEL, 'rb')as f:
        payload=pickle.load(f)
    return payload



def collect_inputs(input_dir):
    """列出输入目录里的数据文件 → [(标识, 完整路径), ...]"""
    files = glob.glob(os.path.join(input_dir, "*.zip"))
    result = []
    for p in sorted(files):
        sid = os.path.basename(p)[:-4]
        result.append((sid, p))
    return result


def split_by_gap(starts, gap_ms):
    """按时间间隔切分 → [(起点下标, 终点下标), ...]"""
    cuts = np.where(np.diff(starts) > gap_ms)[0] + 1
    bounds = np.concatenate([[0], cuts, [len(starts)]])
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def postprocess(prob, starts, params):
    """窗口概率 → 饭段列表 [(起, 止), ...]"""
    thresh = params["THRESH"]
    sigma = params["SIGMA"]
    gap_ms = params["SPLIT_GAP_MS"]
    merge_ms = params["MERGE_GAP_MS"]
    win_ms = (params["WINDOW"] / params["ACC_FS"]) * 1000
    segs_all = []
    for a, b in split_by_gap(starts, gap_ms):
        ps = gaussian_filter1d(prob[a:b], sigma)
        yp = (ps >= thresh).astype(int)
        segs = segment_meals(yp, starts[a:b], win_ms)
        segs = merge_close(segs, merge_ms)
        segs_all.extend(segs)
    return segs_all


def predict_one(clf, params, path):
    """单个输入文件 → 饭段列表"""
    df=load_sensor_zip(path)
    df=df.sort_values("ACC_TIME").reset_index(drop=True)
    t=df["ACC_TIME"].to_numpy()
    seg_list=split_by_gap(t,1000)
    X_list,starts_list=[],[]
    for a,b in seg_list:
        seg=df.iloc[a:b]
        if len(seg)<params["WINDOW"]:
            continue
        feats=[]
        for ax in params["IMU_AXES"]:
            ws=sliding_window(seg[ax].to_numpy(),params["WINDOW"],params["STEP"])
            feats.append(extract_features(ws))
        X_list.append(np.hstack(feats))
        starts_list.append(sliding_window(seg["ACC_TIME"].to_numpy(),params["WINDOW"],params["STEP"])[:,0])
    if not X_list:
        return []

    X = np.vstack(X_list)
    starts = np.concatenate(starts_list)

    prob = clf.predict_proba(X)[:, 1]
    return postprocess(prob, starts, params)
def write_output(rows, out_path):
    """rows = [(受试者ID, 起点ms, 终点ms), ...] → 写成 CSV"""
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "start_ms", "end_ms"])
        w.writerows(rows)


def main():
    payload = load_model()
    clf, params = payload["clf"], payload["params"]

    rows = []
    for sid, path in collect_inputs(INPUT_DIR):
        segs = predict_one(clf, params, path)
        print(f"{sid}: {len(segs)} 段")
        for s in segs:
            rows.append((sid, int(s[0]), int(s[1])))      # ← 注意 int()

    write_output(rows, OUTPUT_CSV)
    print(f"\n共 {len(rows)} 段，已写入 {OUTPUT_CSV}")

    # 双击运行时暂停一下，方便看到输出；被脚本调用（无终端）时直接退出。
    # 注意：PyInstaller 打包后 sys.stdin.isatty() 不可靠，必须 try/except 兜底，
    # 否则无终端环境下 input() 抛 EOFError 会导致非零退出码。
    if getattr(sys, "frozen", False):
        try:
            if sys.stdin.isatty():
                input("\n按回车键退出...")
        except Exception:
            pass


if __name__ == "__main__":
    main()