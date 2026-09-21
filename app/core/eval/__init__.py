"""
评测模块: 数据集、指标、执行。
"""
from app.core.eval.dataset import EvalCase, load_dataset, validate_dataset
from app.core.eval.metrics import (
    aggregate,
    grade_of,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    report_by_kind,
)

__all__ = [
    "EvalCase",
    "load_dataset",
    "validate_dataset",
    "aggregate",
    "grade_of",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "report_by_kind",
]
