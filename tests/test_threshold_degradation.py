# -*- coding: utf-8 -*-
"""重排退化时不得套相似度阈值 —— 两条检索路径的一致性回归（2026-10-17 缺陷）。

缺陷现场（不是假想，是实测）：用生产默认配置问
「被公司辞退了，经济补偿应该怎么算？」—— 知识库里有《劳动合同法》第四十七条，
系统却回答"没有找到足以回答的法律依据"。落库的留痕显示：

    domain_gate(0.3852->high)      ← 拒答闸门是放行的
    rerank(threshold=0.5)          ← 死在这里
    retry#1(0.3) / retry#2(0.2)    ← 三轮全空
    refuse(检索依据不足, 反思得分 0.00)

原因：`ENABLE_RERANK` 关着（或重排器加载失败）时，候选身上的分数是
**RRF 名次倒数之和**（`Σ 1/(60+rank)`，两路都第一也才 0.0328），不是相似度。
拿 0.5 这种"相关度阈值"去卡它，会把全部候选静默清空。

这个防护**早就存在于直连检索**（`retriever.py`，且有 4 条测试守着），但 Agentic
工作流的 `rerank` 节点是**另写一遍的筛选**，漏了这个判断。于是：

  · 生产默认走工作流  → 正常问题一律拒答；
  · 评测脚本走直连    → 52 题全绿，Recall 0.8611，**完全测不到这个 bug**。

所以本文件的重点不是"路 B 要修对"，而是**两条路必须一起对**。
全部脱网可跑。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode

import app.core.workflow.rag_workflow as rw
from app.config.settings import settings
from app.core.rag.retriever import (
    HybridRetriever,
    filter_by_similarity_threshold,
)
from app.core.workflow.rag_workflow import create_rag_workflow, params_for_round

RRF_TOP1 = 0.0328  # 两路都排第一时的真实量级
THRESHOLD = 0.5    # 服务端/请求侧常见的相关度阈值


def _node(text):
    return TextNode(text=text, metadata={"validity_status": "effective"})


# ============================================================
# 1) 规则本身：单位不对时必须跳过
# ============================================================
def test_skips_threshold_when_scores_are_not_relevance():
    """分数不是相关度时, 再低的阈值也不该动它 —— 问题不在阈值高低。"""
    nodes = [NodeWithScore(node=_node("第四十七条"), score=RRF_TOP1)]
    assert filter_by_similarity_threshold(
        nodes, THRESHOLD, scores_are_relevance=False
    ) == nodes
    assert filter_by_similarity_threshold(
        nodes, 0.2, scores_are_relevance=False
    ) == nodes, "降到 0.2 也一样是错的单位, 不能因为它小就放行筛选"


def test_applies_threshold_when_scores_are_relevance():
    """阴性对照: 别把修复做成"永远不过阈值", 那样 0.5 这道质量把关就消失了。"""
    good = NodeWithScore(node=_node("高相关"), score=0.9)
    bad = NodeWithScore(node=_node("低相关"), score=0.1)
    kept = filter_by_similarity_threshold(
        [good, bad], THRESHOLD, scores_are_relevance=True
    )
    assert kept == [good]


def test_no_threshold_keeps_everything():
    a = NodeWithScore(node=_node("a"), score=0.01)
    assert filter_by_similarity_threshold([a], 0.0, scores_are_relevance=True) == [a]
    assert filter_by_similarity_threshold([a], None, scores_are_relevance=True) == [a]


# ============================================================
# 2) rerank_effective 的三种状态
# ============================================================
class _StubVectorRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query):
        return list(self._results)


class _StubIndex:
    def __init__(self, results):
        self.retriever = _StubVectorRetriever(results)

    def as_retriever(self, similarity_top_k=None, filters=None):
        return self.retriever


def _degraded_results():
    return [NodeWithScore(node=_node("第四十七条 每满一年支付一个月工资"), score=RRF_TOP1)]


def test_rerank_effective_is_false_when_reranker_disabled():
    r = HybridRetriever(_StubIndex(_degraded_results()), nodes=[], use_reranker=False)
    assert r.rerank_effective is False


def test_rerank_effective_is_false_when_model_failed_to_load():
    """配置上开着重排、实际没加载成功(内存不足是真场景) —— 与"关闭"同级。"""
    r = HybridRetriever(_StubIndex(_degraded_results()), nodes=[], use_reranker=True)
    r._load_reranker = lambda: None
    r._reranker = None
    assert r.rerank_effective is False


def test_direct_path_keeps_results_when_rerank_degraded():
    """路 A（直连检索）: 退化时结果不许为空。"""
    r = HybridRetriever(_StubIndex(_degraded_results()), nodes=[], use_reranker=True)
    r._load_reranker = lambda: None
    r._reranker = None
    assert len(r.retrieve("经济补偿怎么算", top_k=5, similarity_threshold=THRESHOLD)) == 1


# ============================================================
# 3) 路 B（Agentic 工作流）—— 今天真正坏掉的那条
# ============================================================
class _DegradedRetriever:
    """模拟"重排未生效"的检索器: rerank 原样退回 RRF 名次分。"""

    rerank_effective = False

    def __init__(self, score=RRF_TOP1):
        self._score = score
        self.nodes = [
            NodeWithScore(node=_node("第四十七条 每满一年支付一个月工资"), score=score),
            NodeWithScore(node=_node("第四十六条 应当支付经济补偿的情形"), score=score),
        ]

    def retrieve_raw(self, query):
        return list(self.nodes)

    def vector_top_score(self, query):
        return 0.3852  # 高于 CAUTION(0.36) -> 闸门判 high, 不该拒答

    def rerank(self, query, nodes, top_n=5):
        return nodes[:top_n]

    def get_context_text(self, nodes):
        return "\n".join(n.node.text for n in nodes)

    def get_sources_info(self, nodes):
        return [{"index": i + 1, "node_id": f"n{i}"} for i, n in enumerate(nodes)]


class _ConfidentLLM:
    async def chat(self, messages, **kw):
        return '{"analysis": "片段已足够回答", "confidence": 0.9}'


class _FakeRouter:
    def get(self, name):
        return _ConfidentLLM()


def _run_graph(retriever):
    state = {
        "query": "被公司辞退了，经济补偿应该怎么算？",
        "rewritten_query": "", "hyde_doc": "", "sub_questions": [],
        "retrieved_docs": [], "reranked_docs": [], "context": "",
        "reflection_score": 0.0, "reflection_note": "", "sources": [],
        "retry_count": 0, "retry_params": params_for_round(0),
        "refuse": False, "domain_score": 1.0, "confidence_level": "high",
        "refuse_reason": "", "workflow_steps": [],
    }
    return asyncio.run(
        create_rag_workflow().ainvoke(
            state, config={"configurable": {"retriever": retriever}}
        )
    )


def test_workflow_keeps_results_when_rerank_degraded(monkeypatch):
    """回归本体的缺陷: 重排未生效时, 工作流不得把候选清空、不得拒答。"""
    monkeypatch.setattr(rw, "llm_router", _FakeRouter())
    monkeypatch.setattr(settings, "ENABLE_QUERY_REWRITE", False)
    monkeypatch.setattr(settings, "ENABLE_HYDE", False)
    monkeypatch.setattr(settings, "ENABLE_DOMAIN_GATE", True)

    result = _run_graph(_DegradedRetriever())

    assert not result["refuse"], f"不该拒答: {result.get('refuse_reason')}"
    assert len(result["reranked_docs"]) == 2
    assert result["context"], "参考资料不能为空, 否则生成环节只能编或拒"


def test_workflow_still_applies_threshold_when_rerank_effective(monkeypatch):
    """阴性对照: 重排真生效时阈值照常起作用(不能把修复做成"阈值永远不生效")。

    0.9 的相关度分低于阈值 0.5 -> 应当被筛掉 -> 走拒答。
    """
    monkeypatch.setattr(rw, "llm_router", _FakeRouter())
    monkeypatch.setattr(settings, "ENABLE_QUERY_REWRITE", False)
    monkeypatch.setattr(settings, "ENABLE_HYDE", False)
    monkeypatch.setattr(settings, "ENABLE_DOMAIN_GATE", True)

    degraded_ok = _DegradedRetriever()
    assert not _run_graph(degraded_ok)["refuse"], "前置检查: 同样的分数, 退化时不该拒答"

    class _EffectiveRetriever(_DegradedRetriever):
        rerank_effective = True

    result = _run_graph(_EffectiveRetriever())
    assert result["refuse"], "重排生效且分数低于阈值时, 阈值仍须能把候选筛空"


def test_both_paths_agree_when_rerank_degraded():
    """两条路必须同进同退 —— 这正是本次缺陷的形状: 一条对、一条错, 评测还全绿。"""
    candidates = _degraded_results()

    path_a = HybridRetriever(_StubIndex(candidates), nodes=[], use_reranker=False)
    kept_a = path_a.retrieve("经济补偿怎么算", top_k=5, similarity_threshold=THRESHOLD)

    path_b = filter_by_similarity_threshold(
        candidates, THRESHOLD, scores_are_relevance=_DegradedRetriever.rerank_effective
    )

    assert kept_a, "路 A 不该空"
    assert len(kept_a) == len(path_b), "同一批候选、同一种退化, 两条路结果条数必须一致"
