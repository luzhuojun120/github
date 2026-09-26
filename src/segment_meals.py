


def segment_meals(y_pred,starts,win_ms):
    """把窗口级预测合并成饭段
        返回: [(段起点时刻, 段终点时刻), ...]
    """
    segments=[]
    start_time=None
    for i in range(len(y_pred)):
        if y_pred[i]==1 and (i==0 or y_pred[i-1]==0):
            start_time=starts[i]
        elif y_pred[i]==0 and i>0 and y_pred[i-1]==1:
            segments.append((start_time,starts[i-1]+win_ms))
            start_time=None
        else:
            pass
    if start_time is not None:
       segments.append((start_time,starts[i]+win_ms))
    return segments
def merge_close(segments,gap_ms=180000):
    """

    :param segments:
    :param gap_ms:
    :return:
    """
    if not segments:
        return []
    out=[list(segments[0])]
    for  s,e in segments[1:]:
        if s-out[-1][1]<=gap_ms:
            out[-1][1]=e
        else:
            out.append([s,e])
    return[tuple(x)for x in out]
def filter_short(segments,min_ms=60000):
    """丢掉太短的段（按物理先验：一次进食不会只持续几秒）

        参数
        ----
        segments : [(起, 止), ...]  毫秒时间戳
        min_ms   : int              最短时长；短于它的段被丢弃

        返回
        ----
        [(段起点时刻, 段终点时刻), ...]

        说明
        ----
        阈值只能按物理先验定，不能按 F1 最高定（见 AGENTS.md 约定⑧）。
        60 秒的依据：人一次进食行为至少持续 1 分钟，几秒的波动是噪声。
        实测依据：虚报段时长 4.8s / 7.1s，真饭段最短 64s（P5）。
    """
    return[(s,e)for s,e in segments if (e-s)>=min_ms]
if __name__=='__main__':
    y_pred=[0,0,0,1,1,1,0,0,1,1,0]
    starts=[1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
    win_ms=100
    segs=segment_meals(y_pred,starts,win_ms)
    y_pred2 = [0, 1, 1, 0, 1, 1]
    segs2 = segment_meals(y_pred2, starts[:6], win_ms)
    print("边界测试:", segs2)
    print("检测到饭段：",segs)
    demo=[(0,300),(1000,2500)]
    print("最短过滤 演示：",demo,"最短1000ms →",filter_short(demo,1000))