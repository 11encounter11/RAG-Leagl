"""
评测模块: 数据集、指标、执行。
"""
from app.core.eval.corpus import build_corpus_index, list_laws
from app.core.eval.dataset import DEFAULT_DATASET, EvalCase, load_dataset, validate_dataset
from app.core.eval.metrics import (
    aggregate,
    grade_of,
    grade_hit,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    report_by_kind,
    stale_ratio_at_k,
)

__all__ = [
    "EvalCase",
    "DEFAULT_DATASET",
    "load_dataset",
    "validate_dataset",
    "build_corpus_index",
    "list_laws",
    "aggregate",
    "grade_of",
    "grade_hit",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "report_by_kind",
    "stale_ratio_at_k",
]
