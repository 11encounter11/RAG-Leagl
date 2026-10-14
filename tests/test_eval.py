"""
评测指标与评测集自检测试

为什么指标要测:
    一个写错的指标看着完全合理, 却会误导后续所有优化决策。
    NDCG 里一个手算过的值, 比十次"看起来对了"更有说服力。
"""
from __future__ import annotations

import json

import pytest

from app.core.eval.dataset import EvalCase, load_dataset, validate_dataset
from app.core.eval.metrics import (
    aggregate,
    dcg,
    full_recall_at_k,
    grade_of,
    grade_hit,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    report_by_kind,
)

# ============================================================
# 分档
# ============================================================
def test_grade_of_three_levels():
    required = ["第二十六条"]
    acceptable = ["第二十八条"]
    assert grade_of("第二十六条", required, acceptable) == 2
    assert grade_of("第二十八条", required, acceptable) == 1
    assert grade_of("第五十条", required, acceptable) == 0
    assert grade_of("", required, acceptable) == 0


def test_grade_required_takes_precedence():
    """同一条既是必需又是可接受时, 应按必需算, 不能降档。"""
    assert grade_of("第四十七条", ["第四十七条"], ["第四十七条"]) == 2


# ============================================================
# 法名维度(语料含多部法后必须按法名匹配)
# ============================================================
LABOR = "中华人民共和国劳动法"
CONTRACT = "中华人民共和国劳动合同法"


def test_grade_hit_matches_same_law():
    grade = grade_hit("第三十六条", ["第三十六条"], [], hit_law=LABOR, expected_laws=[LABOR])
    assert grade == 2


def test_grade_hit_rejects_same_article_in_another_law():
    """本测试是整个法名维度的存在理由。

    《劳动法》第三十六条是工时上限, 《劳动合同法》第三十六条是协商解除 ——
    条文号相同、法不同, 绝不能被判成命中。
    """
    grade = grade_hit("第三十六条", ["第三十六条"], [], hit_law=CONTRACT, expected_laws=[LABOR])
    assert grade == 0


def test_grade_hit_treats_missing_law_metadata_as_miss():
    """片段没有法名时不能默认放行 —— 元数据丢失不该算命中。"""
    assert grade_hit("第三十六条", ["第三十六条"], [], hit_law=None, expected_laws=[LABOR]) == 0


def test_grade_hit_accepts_any_of_expected_laws():
    """一个问题的答案可能来自多部法(如社保题), 任一部命中都算对。"""
    grade = grade_hit("第七十二条", ["第七十二条"], [], hit_law=LABOR, expected_laws=[LABOR, CONTRACT])
    assert grade == 2


def test_grade_hit_falls_back_without_expected_laws():
    """未标注法名的样本退回纯条文号匹配(兼容过渡期)。"""
    assert grade_hit("第三十六条", ["第三十六条"], [], hit_law=CONTRACT, expected_laws=[]) == 2


# ============================================================
# DCG / NDCG
# ============================================================
def test_dcg_hand_computed():
    # [0, 3, 0] -> 0/log2(2) + 3/log2(3) + 0/log2(4)
    assert dcg([0, 3, 0]) == pytest.approx(3 / 1.5849625007211562, rel=1e-9)


def test_ndcg_hand_computed():
    """grades=[0,2,0], k=3。

    DCG  = 3/log2(3) = 1.8927892...
    IDCG = 3/log2(2) = 3.0          (理想排序是 [2,0,0])
    NDCG = 1.8927892 / 3 = 0.6309297...
    """
    assert ndcg_at_k([0, 2, 0], 3) == pytest.approx(0.6309297535714574, rel=1e-9)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([2, 2, 0, 0], 4) == pytest.approx(1.0)


def test_ndcg_rewards_better_ordering():
    """同样的命中, 排在前面得分更高 —— 这正是 NDCG 用来衡量重排的原因。"""
    front = ndcg_at_k([2, 0, 0, 0], 4)
    back = ndcg_at_k([0, 0, 0, 2], 4)
    assert front > back
    assert front == pytest.approx(1.0)


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_respects_k():
    """第 3 位的命中在 k=2 时不该被算进去。"""
    assert ndcg_at_k([0, 0, 2], 2) == 0.0
    assert ndcg_at_k([0, 0, 2], 3) > 0.0


# ============================================================
# Recall / MRR / Precision
# ============================================================
def test_recall_counts_any_hit_in_window():
    assert recall_at_k([0, 0, 1, 0], 3) == 1.0
    assert recall_at_k([0, 0, 0, 1], 3) == 0.0
    assert recall_at_k([], 3) == 0.0


def test_full_recall_counts_all_required_articles():
    """复合问题需要两条, 只命中一条只能算一半。"""
    grades = [2, 0, 1, 0]
    assert full_recall_at_k(grades, 4, required_count=2) == pytest.approx(0.5)
    assert full_recall_at_k([2, 0, 2, 0], 4, required_count=2) == pytest.approx(1.0)


def test_reciprocal_rank_uses_first_hit_position():
    assert reciprocal_rank([0, 0, 2, 2]) == pytest.approx(1 / 3)
    assert reciprocal_rank([1, 0, 0]) == pytest.approx(1.0)
    assert reciprocal_rank([0, 0, 0]) == 0.0


def test_precision_at_k_divides_by_window_size():
    assert precision_at_k([2, 0, 1, 0], 4) == pytest.approx(0.5)
    # k 小于结果数时, 分母是 k 而不是结果总数
    assert precision_at_k([2, 2, 0, 0], 2) == pytest.approx(1.0)


def test_metrics_reject_non_positive_k():
    for func in (recall_at_k, ndcg_at_k, precision_at_k):
        with pytest.raises(ValueError):
            func([1], 0)


# ============================================================
# 聚合
# ============================================================
def _item(kind, resolvable, recall=1.0, refused=0.0):
    return {
        "kind": kind,
        "resolvable": resolvable,
        "recall": recall,
        "full_recall": recall,
        "rr": recall,
        "ndcg": recall,
        "precision": recall,
        "refused": refused,
    }


def test_aggregate_excludes_unresolvable_from_recall():
    """超纲题不参与召回指标, 否则"正确答案是召回空"的题会被算成全部漏检。"""
    per_question = [
        _item("term", True, recall=1.0),
        _item("term", True, recall=0.0),
        _item("out_of_scope", False, refused=1.0),
    ]
    result = aggregate(per_question, k=5)
    assert result["resolvable_count"] == 2
    assert result["unresolvable_count"] == 1
    assert result["recall_at_k"] == pytest.approx(0.5)
    assert result["refusal_accuracy"] == pytest.approx(1.0)


def test_aggregate_report_by_kind_isolates_weak_categories():
    per_question = [
        _item("colloquial", True, recall=0.0),
        _item("term", True, recall=1.0),
    ]
    by_kind = report_by_kind(per_question, k=5)
    assert by_kind["colloquial"]["recall_at_k"] == 0.0
    assert by_kind["term"]["recall_at_k"] == 1.0


def test_aggregate_on_empty_input_is_zeros():
    result = aggregate([], k=5)
    assert result["recall_at_k"] == 0.0
    assert result["question_count"] == 0


# ============================================================
# 数据集自检
# ============================================================
def test_real_dataset_loads_and_is_valid():
    """项目实际使用的评测集必须自检通过。"""
    cases = load_dataset("data/eval/legal_qa_eval.jsonl")
    assert len(cases) >= 30, "评测集样本太少, 指标没有统计意义"

    outcome = validate_dataset(cases)
    assert outcome["errors"] == [], outcome["errors"]
    assert outcome["stats"]["resolvable"] >= 20
    assert outcome["stats"]["unresolvable"] >= 3, "必须有超纲题来验证拒答行为"


def test_real_dataset_marks_law_on_every_answerable_case():
    """语料含多部法后, 可回答样本没有 law 就会跨法误判 —— 必须是硬性要求。"""
    cases = load_dataset("data/eval/legal_qa_eval.jsonl")
    missing = [c.id for c in cases if c.answerable and not c.laws]
    assert missing == [], f"这些样本缺 law: {missing}"
    # 超纲题的正确答案是拒答, 不该有 law
    stray = [c.id for c in cases if not c.answerable and c.laws]
    assert stray == [], f"超纲题不该标注 law: {stray}"


def test_dataset_covers_multiple_question_kinds():
    """只测一类问法的评测集测不出真实差距。"""
    cases = load_dataset("data/eval/legal_qa_eval.jsonl")
    kinds = {case.kind for case in cases}
    assert {"term", "colloquial", "out_of_scope"} <= kinds


def test_validate_catches_ground_truth_not_in_corpus():
    """标注时按记忆写条文号, 而语料里根本没有 —— 这是最难发现的标注错误。"""
    cases = [
        EvalCase(
            id="X1",
            question="测试问题",
            kind="term",
            answerable=True,
            laws=[LABOR],
            required=["第九百九十九条"],
        )
    ]
    outcome = validate_dataset(cases, corpus_index={LABOR: {"第一条", "第二条"}})
    assert any("第九百九十九条" in error for error in outcome["errors"])


def test_validate_catches_article_from_wrong_law():
    """条文号写对了、却记错了它属于哪部法 —— 同样必须拦下。"""
    cases = [
        EvalCase(
            id="X3",
            question="测试问题",
            kind="term",
            answerable=True,
            laws=[LABOR],
            required=["第三十六条"],
        )
    ]
    # 第三十六条只在劳动合同法里存在(模拟标注记错法名的情况)
    outcome = validate_dataset(cases, corpus_index={LABOR: {"第一条"}, CONTRACT: {"第三十六条"}})
    assert any("《中华人民共和国劳动法》里没有 第三十六条" in e for e in outcome["errors"])


def test_validate_catches_unknown_law_name():
    cases = [
        EvalCase(
            id="X4",
            question="测试问题",
            kind="term",
            answerable=True,
            laws=["中华人民共和国不存在的法"],
            required=["第一条"],
        )
    ]
    outcome = validate_dataset(cases, corpus_index={LABOR: {"第一条"}})
    assert any("不在语料中" in error for error in outcome["errors"])


def test_validate_requires_law_on_answerable_case():
    cases = [EvalCase(id="X5", question="q", kind="term", answerable=True, required=["第一条"])]
    outcome = validate_dataset(cases, corpus_index={LABOR: {"第一条"}})
    assert any("没标注 law" in error for error in outcome["errors"])


def test_validate_rejects_law_on_out_of_scope_case():
    cases = [
        EvalCase(id="X6", question="q", kind="out_of_scope", answerable=False, laws=[LABOR])
    ]
    outcome = validate_dataset(cases)
    assert any("超纲" in error for error in outcome["errors"])


def test_validate_catches_answerable_without_ground_truth():
    cases = [EvalCase(id="X2", question="没问题", kind="term", answerable=True, required=[])]
    outcome = validate_dataset(cases)
    assert any("没有 required" in error for error in outcome["errors"])


def test_validate_catches_duplicate_ids_and_questions():
    cases = [
        EvalCase(id="A", question="同一个问题", kind="term", answerable=True, required=["第一条"]),
        EvalCase(id="A", question="同一个问题", kind="term", answerable=True, required=["第二条"]),
    ]
    outcome = validate_dataset(cases)
    assert any("id 重复" in error for error in outcome["errors"])
    assert any("问题重复" in warning for warning in outcome["warnings"])


def test_validate_catches_unknown_kind():
    cases = [EvalCase(id="A", question="q", kind="胡说", answerable=True, required=["第一条"])]
    outcome = validate_dataset(cases)
    assert any("未知的 kind" in error for error in outcome["errors"])


def test_dataset_rejects_malformed_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "A", "question": "q"}\n{ 这不是 json }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON 解析失败"):
        load_dataset(bad)


def test_dataset_skips_blank_lines_and_comments(tmp_path):
    path = tmp_path / "ok.jsonl"
    payload = {"id": "A", "question": "q", "kind": "term", "answerable": True, "required": ["第一条"]}
    path.write_text(f"\n# 注释行\n{json.dumps(payload, ensure_ascii=False)}\n\n", encoding="utf-8")
    cases = load_dataset(path)
    assert len(cases) == 1
    assert cases[0].id == "A"
