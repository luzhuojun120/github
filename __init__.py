"""非惯用手进食事件竞赛核心模块。

本入口提供事件级评估接口；数据读取和非惯用手检测接口从各自模块导入。
仅使用评估功能时，无需加载传感器处理模块。
"""

from .metrics import (Event, Match, boundary_delta_from_iou,
                      boundary_expansion_iou, compute_f1, iou, match_events)
from .pipeline import (DataLayer, DecisionLayer, EatingPipeline, FeatureBatch,
                       FeatureLayer, ModelLayer, PipelineConfig,
                       ProbabilitySequence)


def create_server(*args, **kwargs):
    """Lazily create the public HTTP server without importing the CLI module."""
    from .api import create_server as _create_server
    return _create_server(*args, **kwargs)

__all__ = ["Event", "Match", "iou", "match_events", "compute_f1",
           "boundary_expansion_iou", "boundary_delta_from_iou",
           "PipelineConfig", "DataLayer", "FeatureLayer", "ModelLayer",
           "DecisionLayer", "EatingPipeline", "FeatureBatch",
           "ProbabilitySequence", "create_server"]
__version__ = "0.4.0"
