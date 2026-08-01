"""
evaluate.py
===========
评估模块（你主责，最重要）。

作用：
    赛题最终看 F1：把模型预测的进食事件和真实进食事件做"事件级"匹配，
    IoU（交并比）> 0.25 才算正确检测出一个事件，再算灵敏度 / 阳性预测率 / F1。

将来要写的函数：
    - iou(pred_event, true_event): 算两个事件的重叠度
    - match_events(pred_events, true_events): 事件级匹配
    - compute_f1(pred_events, true_events): 输出 F1 / 灵敏度 / 阳性预测率

为什么先做这个：
    数据还没来，但评估逻辑和具体数据无关，现在就能把代码写好、用假数据测通。
"""

# TODO: 实现 iou
# TODO: 实现 match_events
# TODO: 实现 compute_f1
