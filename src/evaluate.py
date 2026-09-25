def iou(seg_pred,seg_true):
    p_start,p_end=seg_pred
    t_start,t_end=seg_true
    inter_start=max(p_start,t_start)
    inter_end=min(p_end,t_end)
    inter_len=max(0,inter_end-inter_start)
    union_start=min(p_start,t_start)
    union_end=max(p_end,t_end)
    union_len=max(0,union_end-union_start)
    if inter_len==0:
        return 0.0
    return inter_len/union_len


def evaluate(pred_segments,true_segments,iou_threshold=0.25):
    """比赛评分：precision / recall / F1
        pred_segments: 模型预测的饭段列表 [(起,止), ...]
        true_segments: 真实饭段列表 [(起,止), ...]
    """
    matched = 0
    for p in pred_segments:
        best=0
        for t in true_segments:
            val=iou(p,t)
            if val>best:
                best=val
        if best>=iou_threshold:
            matched+=1
    precision = matched / len(pred_segments) if pred_segments else 0
    recall = matched / len(true_segments) if true_segments else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision,recall,f1

if __name__ == "__main__":
    # --- iou 自测 ---
    print(iou((100, 200), (100, 200)))
    print(iou((100, 200), (300, 400)))
    print(iou((100, 200), (150, 250)))

    # --- evaluate 自测 ---
    pred = [(1300, 1600), (1800, 2000), (5000, 5200)]      # 预测3段（最后一段是误报）
    true = [(1300, 1600), (1800, 2000), (7000, 7300)]      # 真实3段
    precision, recall, f1 = evaluate(pred, true)
    print(f"precision={precision:.2f} recall={recall:.2f} F1={f1:.2f}")




