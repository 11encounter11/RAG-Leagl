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

# 与 app/core/ingest/metadata.VALIDITY_EFFECTIVE 必须一致。
# 这里不 import 那个常量, 是因为 `import app.core.ingest.metadata` 会先执行
# `app/core/ingest/__init__.py`, 把 llama-index 整个拉进来 —— 本模块刻意保持
# 零重依赖(纯函数, 可单独测)。两处会不会漂移由测试盯着:
# tests/test_eval.py::test_stale_ratio_agrees_with_metadata_constant
EFFECTIVE_STATUS = "effective"


def grade_of(article_no: str, required: Sequence[str], acceptable: Sequence[str]) -> int:
    """把一条召回结果映射成相关性分档。"""
    if article_no and article_no in required:
        return 2
    if article_no and article_no in acceptable:
        return 1
    return 0


def grade_hit(
    article_no: str,
    required: Sequence[str],
    acceptable: Sequence[str],
    *,
    hit_law: str | None = None,
    expected_laws: Sequence[str] | None = None,
) -> int:
    """带法名的分档 —— 语料里有多部法时**必须**用它, 而不是 grade_of。

    为什么需要: 每部法都有自己的"第三十六条"。语料只有一部法时, 单看条文号
    没有歧义; 一旦语料里有《劳动法》和《劳动合同法》, 一条"第三十六条"既可能是
    劳动法的工时上限, 也可能是劳动合同法的协商解除。只按条文号匹配会把
    **另一部法的同名条文**判成命中, 指标虚高且不报错。

    规则:
        - `expected_laws` 为空 -> 退回纯条文号匹配(兼容未标注法名的样本)
        - `expected_laws` 非空 -> 命中片段的法名必须落在其中, 否则判 0
          (片段没有法名也算不匹配 —— 元数据缺失不该被当成命中)
    """
    if expected_laws:
        if not hit_law or hit_law not in expected_laws:
            return 0
    return grade_of(article_no, required, acceptable)


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


def stale_ratio_at_k(statuses: Sequence[str | None], k: int) -> float:
    """旧版混入率: top-k 里"非现行有效"片段所占比例。

    为什么必须单独有这个指标(2026-10-15 补):
        时效过滤的价值是"不引用已修订/已废止的条文", 而不是"召回率更高"。
        拿召回率衡量它, 甚至会得出相反的结论 —— 开了过滤指标反而下降,
        因为 ground truth 只写"第X条"、不区分版本, 靠旧版本命中也被算对。
        这个指标直接数混进来的旧版本: 开过滤应当为 0, 关过滤应当 > 0。
        它才是 T-28"开关对照"可以量化的证据。

    判据: 只要 `validity_status` 不是 `effective` 就算混入 ——
    amended / repealed / pending 显然算, **空值也算**: 无法证明它是现行版本时,
    在"回答可信"这件事上和旧版一样危险, 不该被悄悄放过。

    分母是**实际取到的**片段数(不足 k 时按实际算), 不是固定 k:
    少召回和召回到旧版是两个问题, 按固定 k 算会让前者冒充后者。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    window = list(statuses[:k])
    if not window:
        return 0.0
    return sum(1 for status in window if status != EFFECTIVE_STATUS) / len(window)


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
        # 时效过滤的专用指标: 见 stale_ratio_at_k 的说明
        "stale_ratio_at_k": mean([item["stale_ratio"] for item in resolvable]),
        "questions_with_stale": sum(1 for item in resolvable if item["stale_ratio"] > 0),
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
