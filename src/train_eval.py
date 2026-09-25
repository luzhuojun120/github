"""train_eval.py —— 按受试者切分的训练与评估"""
import os
import glob
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.ndimage import gaussian_filter1d
from segment_meals import segment_meals, merge_close
from evaluate import evaluate,iou
WINDOW = 500
ACC_FS = 105
WINDOW_MS = (WINDOW / ACC_FS) * 1000




def load_all_cache(cache_dir,min_pos=100):
    """读全部受试者缓存，只保留「惯用手场景」的窗口。

        参数
        ----
        cache_dir : str   缓存目录（data/cache）
        min_pos   : int   正样本少于这个数的受试者直接排除

        返回
        ----
        dict: {受试者ID: {"X": (n,30) float, "y": (n,) 0/1}}
        """

    out={}
    files=glob.glob(os.path.join(cache_dir,"*.npz"))
    for p in files:
        sid=os.path.basename(p)[:-4]
        d=np.load(p)
        X=d["X"]
        is_dominant=d["is_dominant"]
        meal_labels =d["meal_labels"]
        starts = d["starts"]
        pos = (( is_dominant& (meal_labels == 1)).sum())
        if pos < min_pos:
            continue
        out[sid]={"X": X[is_dominant],"y" : (meal_labels[is_dominant] == 1).astype(int),"starts":starts[is_dominant],}
    return out
def split_subject(data,test_ratio=0.2,seed=42):
    """

    :param data:
    :param test_ratio:
    :param seed:
    :return:
    """
    ids=list(data.keys())
    random.Random(seed).shuffle(ids)
    n_test=int(len(ids)*test_ratio)
    train_ids=ids[:-n_test]
    test_ids=ids[-n_test:]
    return train_ids,test_ids
def train_and_eval(data,train_ids,test_ids):
    """

    :param data:
    :param train_ids:
    :param test_ids:
    :return:
    """
    X_tr=np.vstack([data[i]["X"]for i in train_ids])
    y_tr=np.concatenate([data[i]["y"]for i in train_ids])
    X_te=np.vstack([data[i]["X"]for i in test_ids])
    y_te=np.concatenate([data[i]["y"]for i in test_ids])
    clf=RandomForestClassifier(n_estimators=100,class_weight="balanced",random_state=42,n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred=clf.predict(X_te)
    p=precision_score(y_te,y_pred)
    r=recall_score(y_te,y_pred)
    f1=f1_score(y_te,y_pred)
    return clf, p,r,f1
def eval_event(clf,sub,thresh=0.4,sigma=3,gap_ms=300000):
    prob=clf.predict_proba(sub["X"])[:,1]
    starts=sub["starts"]
    y_true=sub["y"]
    cuts=np.where(np.diff(starts)>gap_ms)[0]+1
    bounds=np.concatenate([[0],cuts,[len(starts)]])
    ranges=[(bounds[i],bounds[i+1])for i in range(len(bounds)-1)]
    pred_segs,true_segs=[],[]
    for a,b in ranges:
        p=prob[a:b]
        ps=gaussian_filter1d(p,sigma)
        yp=(ps>=thresh).astype(int)
        segs=segment_meals(yp,starts[a:b],WINDOW_MS)
        segs=merge_close(segs,180000)
        pred_segs.extend(segs)
        yt=y_true[a:b]
        segs=segment_meals(yt,starts[a:b],WINDOW_MS)
        true_segs.extend(segs)
    return pred_segs,true_segs
def match_count(pred_segs,true_segs,thresh=0.25):
    """

    :param pred_segs:
    :param true_segs:
    :param thresh:
    :return:
    """
    used=set()
    m=0
    for p in pred_segs:
        best_i,best_v=None,0.0
        for i,t in enumerate(true_segs):
            if i in used:
                continue
            v=iou(p,t)
            if v>best_v:
                best_v=v
                best_i=i
        if best_v>=thresh:
            used.add(best_i)
            m+=1
    return m


if __name__=="__main__":
    from paths import CACHE
    data=load_all_cache(CACHE)
    print("可用受试者，",len(data))
    print("总窗口：",sum(d["X"].shape[0]for d in data.values()))
    print("总正样本：",sum(int(d["y"].sum())for d in data.values()))
    tr,te=split_subject(data)
    print("训练组：",len(tr),"|测试组：",len(te),"人")
    print("交集：",set(tr)&set(te))
    print("训练正样本：",sum(int(data[i]["y"].sum())for i in tr))
    print("测试正样本：", sum(int(data[i]["y"].sum()) for i in te))
    clf, p,r,f1=train_and_eval(data,tr,te)
    print(f"precision={p:.4f}recall={r:.4f}f1={f1:.4f}")
    te0=te[0]
    prob=clf.predict_proba(data[te0]["X"])[:,1]
    print("受试者：",te0)
    print("prob形状：",prob.shape)
    print("min/max/mean：",prob.min(),prob.max(),prob.mean())
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(th, (prob >= th).sum(), (prob >= th).mean())
    y=data[te0]["y"]
    print("真实正比例：",y.mean(),"|正样本数：",y.sum(),"总窗口：",len(y))
    st=data[te0]["starts"]
    dif=np.diff(st)
    print("相邻间隔中位数：",np.median(dif),"最小：",dif.min(),"最大：",dif.max())
    print("starts 是否单调递增:", bool(np.all(np.diff(st) > 0)))
    print("间隔 > 10 秒的断点数:", int((np.diff(st) > 10000).sum()))
    big = np.diff(st)[np.diff(st) > 10000]
    print("这些大间隔:", big[:20])
    ps, ts = eval_event(clf, data[te0])
    print("预测段数:", len(ps), "| 真值段数:", len(ts))
    print("预测前3段:", ps[:3])
    print("真值前3段:", ts[:3])
    tp_all = fp_all = fn_all = 0
    for sid in te:
        ps, ts = eval_event(clf, data[sid])
        mc = match_count(ps, ts)
        tp_all += mc
        fp_all += len(ps)-mc
        fn_all += len(ts)-mc
        print(f"{sid}: 预测{len(ps)}段 | 真值{len(ts)}段 | 命中{mc}")
    P = tp_all / (tp_all + fp_all) if (tp_all + fn_all) else 0
    R = tp_all / (tp_all + fn_all)if (tp_all+fn_all)else 0                                 # ← 你填
    F =    2*P*R/(P+R)if (P+R) else 0
    print(f"7人微平均: P={P:.4f} R={R:.4f} F1={F:.4f}")
    sid = "HNU21018"
    sub = data[sid]
    pr = clf.predict_proba(sub["X"])[:, 1]
    print(f"{sid} prob: min={pr.min():.4f} max={pr.max():.4f} mean={pr.mean():.4f}")
    for th in [0.2, 0.3, 0.4, 0.5]:
        print(f"  prob>={th}: {(pr >= th).sum()} 个窗口")
    print(f"  真值正样本: {int(sub['y'].sum())} / {len(sub['y'])}")

    st = sub["starts"]
    cuts = np.where(np.diff(st) > 300000)[0] + 1
    print(f"  按 5 分钟切开: {len(cuts)+1} 段")
    bounds = np.concatenate([[0], cuts, [len(st)]])
    lens = [bounds[i+1]-bounds[i] for i in range(len(bounds)-1)]
    print(f"  各段长度: 最小={min(lens)} 最大={max(lens)} 中位数={int(np.median(lens))}")
    print(f"  长度 < 500 的段数: {sum(1 for L in lens if L < 500)}")




