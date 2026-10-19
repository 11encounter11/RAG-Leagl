# -*- coding: utf-8 -*-
"""多路查询合并与重排问句的回归测试（2026-10-19 缺陷）。

缺陷现场（实测，不是推演）：问「公司一直不给我交社保，我该怎么办？」

    《社会保险法》第六十三条
      · 用原始口语问题检索      -> 候选池 30 条里排第 11
      · 用 LLM 改写后的句子检索 -> 候选池 26 条里排第 **1**
      · 走完整 Agentic 工作流   -> 仍然**拒答**（反思 0.20，重试两轮用尽）

两个原因叠在一起：

1. 合并用的是「按 node_id 保留首次出现」。原问题先跑，第六十三条就在合并名单里
   占了第 11 位；改写那一轮它的第 1 名因为是重复项被**整条丢弃**。
2. 即使重排器开着也救不了：`rerank` 节点拿的是 `state["query"]`（原始口语句）去给
   「按改写句召回来的候选」打分 —— 换一个问题评判上一问题的结果，改写捞回来的
   高分条文被系统性压低。

净效果：**改写节点每次都在白烧一次 LLM 调用**，而日志上完全看不出来。

本文件守三件事：跨查询合并必须用同一套 RRF、重排必须按改写句打分、
以及"同一个算法不许有两份实现"。全部脱网可跑。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode

import app.core.rag.retriever as retriever_module
import app.core.workflow.rag_workflow as rw
from app.config.settings import settings
from app.core.rag.retriever import reciprocal_rank_fusion
from app.core.workflow.rag_workflow import create_rag_workflow, params_for_round

TARGET_ARTICLE = "第六十三条　劳动争议申请仲裁的时效期间为一年。"  # 充当"改写后才浮上来的那条"
NOISE = "其他条文内容。"


def _node(nid: str, text: str = NOISE) -> NodeWithScore:
    # 融合用的是 TextNode.node_id (不是 metadata 里的同名键), 桩必须设在真字段上,
    # 否则所有节点 node_id 都是 "" -> 互相碰撞, 测的就不是合并而是去重了。
    return NodeWithScore(
        node=TextNode(id_=nid, text=text, metadata={"validity_status": "effective"}),
        score=0.5,
    )


def _ids(nodes) -> list[str]:
    return [n.node.node_id for n in nodes]


# ============================================================
# 1) 算法本身: 两路都靠前才算靠前
# ============================================================
def test_document_ranked_first_in_one_path_rises_to_top():
    """核心回归: 某条只在一路上排第 1, 不该被"首次出现顺序"压到第 11 位。"""
    path_original = [_node(f"a{i}") for i in range(10)] + [_node("target")]  # target 第 11
    path_rewritten = [_node("target")] + [_node(f"b{i}") for i in range(9)]  # target 第 1

    merged = _ids(reciprocal_rank_fusion(path_original, path_rewritten))
    assert merged[0] == "target", f"合并后 target 在第 {merged.index('target') + 1} 位"

    # 对照旧实现(按 node_id 保留首次出现): target 会留在第 11 位
    seen, first_occurrence = set(), []
    for n in [*path_original, *path_rewritten]:
        nid = n.node.node_id
        if nid not in seen:
            seen.add(nid)
            first_occurrence.append(nid)
    assert first_occurrence.index("target") == 10, (
        "旧算法在这里应当把 target 压在第 11 位 —— 这条断言是用来证明差异真实存在的"
    )


def test_consensus_ranking_beats_single_path_ranking():
    """两路都在第 3 的, 应该压过只在一路排第 1 的(这才是 RRF 的本意)。"""
    both_third = _node("both")
    path_a = [_node("a1"), _node("a2"), both_third]
    path_b = [_node("b1"), _node("b2"), both_third]
    only_a_first = _node("solo")
    merged = _ids(reciprocal_rank_fusion(path_a, path_b, [only_a_first] + [both_third]))
    assert merged[0] == "both"


# ============================================================
# 2) 真实代码路径: 工作流的合并与重排
# ============================================================
class RecordingRetriever:
    """按查询文本给不同排序, 用来模拟"改写后才把目标条文捞上来"。"""

    # 本桩的 rerank() 只是截断, 不产生相关度分 —— 所以诚实地报"重排未生效",
    # 这正是 L21 被实测时的环境(ENABLE_RERANK=false)。
    # 若谎称 True, 阈值会拿 RRF 分(0.03)去卡, 测的就是别的东西了。
    rerank_effective = False

    def __init__(self):
        self.rerank_queries: list[str] = []

    def retrieve_raw(self, query):
        if "社会保险法" in query or "补缴" in query:  # 改写句
            return [_node("target", TARGET_ARTICLE)] + [_node(f"w{i}") for i in range(9)]
        return [_node(f"o{i}") for i in range(10)] + [_node("target", TARGET_ARTICLE)]

    def vector_top_score(self, query):
        return 0.40  # 高于低置信线, 闸门放行

    def rerank(self, query, nodes, top_n=5):
        self.rerank_queries.append(query)
        return nodes[:top_n]

    def get_context_text(self, nodes):
        return "\n".join(n.node.text for n in nodes)

    def get_sources_info(self, nodes):
        return [{"index": i + 1, "node_id": n.node.node_id} for i, n in enumerate(nodes)]


class RewritingLLM:
    REWRITE_MARK = "改写成更适合在法律"
    model_name = "stub-rewrite-model"

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, **kw):
        self.calls += 1
        prompt = messages[-1]["content"]
        if self.REWRITE_MARK in prompt:
            return "用人单位未依法缴纳社会保险费，劳动者能否要求补缴社保、向社保征收机构投诉举报。"
        return '{"analysis": "片段已足够回答", "confidence": 0.9}'


class _Router:
    def __init__(self, llm):
        self._llm = llm

    def get(self, name):
        return self._llm


class _NoCacheRedis:
    """空缓存假件。

    不拦掉的话, 这几个测试会去读真实 Redis —— 里面存着之前手工跑 L21 时留下的
    改写结果, 那会让"用哪句改写句检索"变成环境相关, 测试就不再测同一段逻辑了。
    """

    async def cache_get(self, key):
        return None

    async def cache_set(self, key, value, ttl=None):
        return None


def _state(query):
    return {
        "query": query, "rewritten_query": "", "hyde_doc": "", "sub_questions": [],
        "retrieved_docs": [], "reranked_docs": [], "context": "",
        "reflection_score": 0.0, "reflection_note": "", "sources": [],
        "retry_count": 0, "retry_params": params_for_round(0),
        "refuse": False, "domain_score": 1.0, "confidence_level": "high",
        "refuse_reason": "", "workflow_steps": [],
    }


def _run(monkeypatch, retriever, llm):
    monkeypatch.setattr(rw, "llm_router", _Router(llm))
    monkeypatch.setattr(rw, "redis_store", _NoCacheRedis(), raising=True)
    monkeypatch.setattr(settings, "ENABLE_QUERY_REWRITE", True)
    monkeypatch.setattr(settings, "ENABLE_HYDE", False)
    monkeypatch.setattr(settings, "ENABLE_DOMAIN_GATE", True)
    monkeypatch.setattr(settings, "ENABLE_RERANK", True)
    return asyncio.run(
        create_rag_workflow().ainvoke(
            _state("公司一直不给我交社保，我该怎么办？"),
            config={"configurable": {"retriever": retriever}},
        )
    )


def test_workflow_keeps_target_from_rewritten_path(monkeypatch):
    """走完真实工作流: 改写句里排第 1 的那条, 必须真的进到最终候选。"""
    retriever, llm = RecordingRetriever(), RewritingLLM()
    result = _run(monkeypatch, retriever, llm)

    assert not result["refuse"], f"不该拒答: {result.get('refuse_reason')}"
    assert "target" in _ids(result["reranked_docs"]), (
        f"目标条文被合并策略丢了: {_ids(result['reranked_docs'])}"
    )


def test_rerank_scores_against_rewritten_query_not_original(monkeypatch):
    """重排不许再用原始口语句给"按改写句召回的候选"打分。"""
    retriever, llm = RecordingRetriever(), RewritingLLM()
    _run(monkeypatch, retriever, llm)

    assert retriever.rerank_queries, "重排节点没被调用"
    for q in retriever.rerank_queries:
        assert "公司一直不给我交社保" not in q, "重排仍在用原始口语问题打分"
        assert "社会保险法" in q or "补缴" in q


def test_workflow_does_not_reimplement_fusion():
    """锁"一个算法只有一份实现"。

    本项目栽过好几次同一个形状: 同一条规则在两处各写一遍, 改一处忘另一处
    (时效过滤只加在向量那路、阈值防护只在直连检索、多路合并自写一套去重)。
    这条断言让"工作流自己又写一个融合"直接失败。
    """
    assert rw.reciprocal_rank_fusion is retriever_module.reciprocal_rank_fusion
    assert rw.filter_by_similarity_threshold is retriever_module.filter_by_similarity_threshold


def test_threshold_log_prints_enough_digits(monkeypatch):
    """闸门余量只有 0.0105, 日志把 0.345 印成 0.34 会在诊断时误导人。"""
    import inspect

    source = inspect.getsource(rw.create_rag_workflow)
    assert "拒答线 %.2f" not in source, "阈值日志精度不够, 0.345 会被印成 0.34"
