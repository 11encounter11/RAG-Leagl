"""
评测指标 —— 纯函数, 不依赖任何基础设施。

为什么指标要单独成一个模块并配单元测试:
    评测脚本本身出错的方式非常隐蔽 —— 一个索引差一、一个分母写错,
    出来的数字看着合理但全是错的, 而"看起来合理的错数字"比没有数字更危险
    (会拿它当基线去指导优化)。所以指标必须用构造数据验证过才准用。

相关性分档(gain):
    2 = 必需命中(该问题必须检索到的那一条法条)
    1 = 可接受(同类问题、相邻条款, 命中算部分正确)
    0 = 无关
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def grade_of(article_no: str, required: Sequence[str], acceptable: Sequence[str]) -> int:
    """把一条召回结果映射成相关性分档。"""
    if article_no and article_no in required:
        return 2
    if article_no and article_no in acceptable:
        return 1
    return 0


def dcg(gains: Iterable[float]) -> float:
    """折损累计增益。位置 i(从 0 开始)的折损系数是 log2(i + 2)。"""
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def recall_at_k(grades: Sequence[int], k: int) -> float:
    """命中率: top-k 里是否至少有一条相关内容。

    二值指标, 只回答"有没有找到", 不回答"排得好不好"。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if not grades:
        return 0.0
    return 1.0 if any(grade > 0 for grade in grades[:k]) else 0.0


def full_recall_at_k(grades: Sequence[int], k: int, required_count: int) -> float:
    """完整召回率: top-k 覆盖了多少比例的必需条。

    与 recall_at_k 的区别: 复合问题需要不止一条法条, 只命中一条不算完整。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if required_count <= 0:
        return 1.0 if any(g > 0 for g in grades[:k]) else 0.0
    hits = sum(1 for grade in grades[:k] if grade >= 2)
    return min(1.0, hits / required_count)


def reciprocal_rank(grades: Sequence[int]) -> float:
    """第一条相关内容出现位置的倒数。0 表示全都没命中。"""
    for index, grade in enumerate(grades):
        if grade > 0:
            return 1.0 / (index + 1)
    return 0.0


def ndcg_at_k(grades: Sequence[int], k: int) -> float:
    """归一化折损累计增益。

    比命中率更能反映排序质量: 把答案从第 1 位挪到第 5 位, 命中率不变, NDCG 会掉。
    这也是它能同时反映"重排有没有起作用"的原因。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if not grades:
        return 0.0
    gains = [2**grade - 1 for grade in grades[:k]]
    ideal = sorted((2**grade - 1 for grade in grades), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def precision_at_k(grades: Sequence[int], k: int) -> float:
    """top-k 里相关条目所占比例。"""
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if not grades:
        return 0.0
    window = grades[:k]
    return sum(1 for grade in window if grade > 0) / len(window)


def aggregate(per_question: List[dict], k: int) -> dict:
    """把逐题结果聚合成总体指标。

    只对**有标准答案**的题(resolvable=True)计算召回类指标; 无答案题
    (超纲题)单独统计拒答表现 —— 它们的正确答案是"什么都不召回",
    混在一起算平均会让指标失去意义。
    """
    resolvable = [item for item in per_question if item.get("resolvable")]
    unresolvable = [item for item in per_question if not item.get("resolvable")]

    def mean(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    report = {
        "k": k,
        "question_count": len(per_question),
        "resolvable_count": len(resolvable),
        "unresolvable_count": len(unresolvable),
        "recall_at_k": mean([item["recall"] for item in resolvable]),
        "full_recall_at_k": mean([item["full_recall"] for item in resolvable]),
        "mrr": mean([item["rr"] for item in resolvable]),
        "ndcg_at_k": mean([item["ndcg"] for item in resolvable]),
        "precision_at_k": mean([item["precision"] for item in resolvable]),
        # 超纲题: 检索返回空才算通过
        "refusal_accuracy": mean([item["refused"] for item in unresolvable]),
    }
    return report


def report_by_kind(per_question: List[dict], k: int) -> dict:
    """按问题类型分组统计 —— 用来定位"哪一类问法效果差"。"""
    kinds = sorted({item["kind"] for item in per_question})
    return {
        kind: aggregate([item for item in per_question if item["kind"] == kind], k)
        for kind in kinds
    }
