"""detect_dominant.py---惯用手（IMU6轴）进食检测"""
import sys,os


import numpy as np
import datetime
from scipy.ndimage import gaussian_filter1d


sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_sensor_zip
from preprocess import sliding_window,make_labels
from features import window_features,extract_features,ACC_FS
from sklearn.ensemble import RandomForestClassifier
from segment_meals import segment_meals,merge_close
from paths import RAW, BASE

IMU_AXES=["ACC_X","ACC_Z","ACC_Y","GYRO_X","GYRO_Y","GYRO_Z"]
WINDOW,STEP=500,250


def extract_imu_features(window):
    pass

def split_segments(df,gap_ms=1000):
    """

    :param df:
    :param gap_ms:
    :return:
    """
    t=df["ACC_TIME"].to_numpy()
    breaks=np.where(np.diff(t)>gap_ms)[0]+1
    bound=np.concatenate([[0],breaks,[len(df)]])
    return [(bound[i],bound[i+1])for i in range(len(bound)-1)]
def build_imu_matrix(df):
    all_feats = []
    for ax in IMU_AXES:
        data = df[ax].to_numpy()
        windows = sliding_window(data, WINDOW, STEP)
        feats = extract_features(windows, ACC_FS)
        all_feats.append(feats)
    t = df['ACC_TIME'].to_numpy()
    starts = sliding_window(t, WINDOW, STEP)[:, 0]
    return np.hstack(all_feats),starts
def build_imu_matrix_segment(df,gap_ms=1000):
    """

    :param df:
    :param gap_ms:
    :return:
    """
    all_X,all_starts=[],[]
    for a,b in split_segments(df,gap_ms):
        seg=df.iloc[a:b]
        if len(seg)<WINDOW:
              continue
        Xi, si = build_imu_matrix(seg)
        all_X.append(Xi)
        all_starts.append(si)
    return np.vstack(all_X),np.concatenate(all_starts)
def detect_dominant(train_df,train_meals,test_df,test_meals):

    win_ms=(WINDOW/ACC_FS)*1000
    X_tr,st_tr=build_imu_matrix_segment(train_df)
    y_tr=make_labels(st_tr,win_ms,train_meals)
    print("训练集采集正样本比例：",round(float(y_tr.mean()),4),"|个数：",int(y_tr.sum()),"/",len(y_tr))

    #X,starts=build_imu_matrix_segment(df)
    #y=make_labels(starts,win_ms,meals)
    clf=RandomForestClassifier(n_estimators=100,random_state=42)
    clf.fit(X_tr,y_tr)
    print("训练集上预测为1的比例：",round(float(clf.predict(X_tr).mean()),4))

    
    # ② 测试：提特征 → 概率 → 平滑 → 阈值
    X_te,st_te=build_imu_matrix_segment(test_df)
    proba=clf.predict_proba(X_te)[:,1]
    smooth=gaussian_filter1d(proba,sigma=5)
    pred=(smooth>0.3).astype(int)
    print("训练集上实际为1的比例：",round(float(pred.mean()),4))

    # ③ 窗口预测 → 饭段 → 合并
    segments=segment_meals(pred,st_te,win_ms)
    print("合并前：", len(segments))
    segments=merge_close(segments,gap_ms=180000)
    print("合并后：", len(segments))

    return segments




if __name__=="__main__":
    import glob
    import pandas as pd
    from data_loader import load_sensor_zip
    SUBJECT="HNU21026"
    meal_df=pd.read_csv(os.path.join(RAW,"mealinfo_标注表.csv"),encoding="utf-8-sig")
    rows=meal_df[meal_df["externalid"]==SUBJECT]
    meals=list(zip(rows["beforeTime"],rows["afterTime"]))
    print("饭段数：",len(meals))
    map_df=pd.read_csv(os.path.join(RAW,"sensor_下载映射表.csv"),encoding="utf-8-sig")
    hnu=map_df[map_df["externalid"]==SUBJECT]
    zip_files=hnu["sensorData"].tolist()
    zip_starts=hnu["timeStamp.startTime"].tolist()
    zip_ends=hnu["timeStamp.endTime"].tolist()
    print("zip数",len(zip_files))
    dfs=[]
    for f in zip_files:
        p=os.path.join(RAW,"sensorData",os.path.basename(f))
        dfs.append(load_sensor_zip(p))
    print("读入表数：",len(dfs),"|各表行数合计：",sum(len(d)for d in dfs))
    big=pd.concat(dfs,ignore_index=True)
    big=big.sort_values("ACC_TIME").reset_index(drop=True)
    print("拼接后行数：",len(big))
    print("开始时间",big["ACC_TIME"].iloc[0],"|末行时间：",big["ACC_TIME"].iloc[-1])
    seg_idx=split_segments(big)
    print("连续行数：",len(seg_idx))
    print("段总行数：",sum(b-a for a,b in seg_idx),"|原始行数：",len(big))
    #X,starts=build_imu_matrix_segment(big)
    #print("窗口数：",len(starts),"|X:",X.shape)
    #d=np.diff(starts)
   # print("段边界跳变数:",int((d>5000).sum()))
    mid=big["ACC_TIME"].iloc[len(big)//2]
    train_df=big[big["ACC_TIME"]<mid]
    test_df=big[big["ACC_TIME"]>=mid]
    train_meals=[(c,d)for (c,d)in meals if d<=mid]
    test_meals=[(c,d)for (c,d)in meals if c>=mid]
    print("训练饭段：",len(train_meals),"|测试饭段：",len(test_meals))
    segs = detect_dominant(train_df=train_df, train_meals=train_meals,test_df=test_df,   test_meals=test_meals)
    print("检出饭段：",len(segs))
    print(segs)
    cache=os.path.join(BASE,"data","cache_X_starts.npz")
    if os.path.exists(cache):
        d=np.load(cache)
        X,starts=d["X"],d["starts"]
        print("从缓存读取：",X.shape)
    else:
        X,starts=build_imu_matrix_segment(big)
        np.savez_compressed(cache,X=X,starts=starts)
        print("已经计算并缓存：",X.shape)
    from evaluate import evaluate
    p,r,f1=evaluate(segs,test_meals)
    print("precision:",p,"recall:",r,"F1:",f1)





    np.savez_compressed(os.path.join(BASE,"data","cache_X_starts.npz"),
                        X=X, starts=starts)
    print("已缓存")
        #best_indx=0
    #best_cnt=-1
    #for i in range(len(zip_files)):
        #cnt=sum(1 for (c,d)in meals
         #       if c>zip_starts[i] and d<zip_ends[i])
        #print(f"{i}覆盖{cnt}顿")
        #if cnt>best_cnt:
         #   best_cnt=cnt
          #  best_indx=i
    #chosen=os.path.basename(zip_files[best_indx])
    #print("选中",chosen,"|覆盖",best_cnt,"顿")
    #df=load_sensor_zip(os.path.join(RAW,"sensorData",chosen))



    #folder=r"E:\workbuddy\进食检测比赛\data\raw\sensorData"
    #zips=glob.glob(folder+r"\*.zip")
    #print("找到zip数量：",len(zips))
    #print("第一个：",zips[0][-40:])
    #df=load_sensor_zip(zips[0])
    #print("读入行数：",len(df))
    #win_ms=(WINDOW/ACC_FS)*1000
    #meals=[(starts[100],starts[200])]
    #segs=detect_dominant(df,meals)
    #if segs:
     #   print("饭段起点时间：",datetime.datetime.fromtimestamp(segs[0][0]/1000))
      #  print("饭段终点时间：",datetime.datetime.fromtimestamp(segs[0][1]/1000))
    #print(datetime.datetime.fromtimestamp(1784472863338/1000))
    #print("检测出饭段：",len(segs))
    #print("检出饭段：",segs)

    #rint("假饭段：",(starts[100],starts[199]+win_ms))





