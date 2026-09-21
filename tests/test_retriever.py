"""
检索器与引用格式测试

对应验收:
    T-01 引用格式统一
    T-02 重排分数归一化 / 阈值语义
    T-06 召回与重排分离

设计原则: 全部使用桩对象, 不依赖 Milvus / MySQL / Redis, 也不下载真实的
Cross-Encoder 模型 —— 单元测试必须能在没有基础设施的机器上跑。
"""
from __future__ import annotations

import math

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.core.generate.citation import (
    CITATION_INDEX_FORMAT,
    build_context_block,
    build_sources,
    format_citation_label,
)
from app.core.rag.retriever import (
    HybridRetriever,
    normalize_rerank_score,
    tokenize_zh,
)


# ============================================================
# 桩对象
# ============================================================
class _StubVectorRetriever:
    def __init__(self, results):
        self._results = results
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        return list(self._results)


class _StubIndex:
    def __init__(self, results):
        self._results = results
        self.retriever = _StubVectorRetriever(results)

    def as_retriever(self, similarity_top_k=None):
        return self.retriever


class _FakeReranker:
    """模拟 Cross-Encoder: 返回原始 logits(可正可负), 而不是 0~1 概率。"""

    def __init__(self, logits_by_text=None):
        self._logits = logits_by_text or {}
        self.calls = 0
        self.last_candidate_count = 0

    def postprocess_nodes(self, nodes, query_str=None):
        self.calls += 1
        self.last_candidate_count = len(nodes)
        scored = []
        for index, node in enumerate(nodes):
            raw = self._logits.get(node.node.text, float(len(nodes) - index))
            scored.append(NodeWithScore(node=node.node, score=raw))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored


def _node(text: str, **metadata) -> TextNode:
    return TextNode(text=text, metadata=metadata)


def _build(results, nodes_by_id=None):
    index = _StubIndex(results)
    retriever = HybridRetriever(index, nodes=nodes_by_id or [], use_reranker=True)
    return retriever, index


# ============================================================
# T-02 分数归一化
# ============================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.0, 0.0),
        (0.8, 0.8),
        (1.0, 1.0),
        (3.0, 1.0),          # 默认按概率处理: 越界只钳制, 保持单调
        (-3.0, 0.0),
        (None, 0.0),
    ],
)
def test_normalize_rerank_score_probability_mode(raw, expected):
    assert round(normalize_rerank_score(raw, treat_as_logit=False), 4) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.0, 0.5),
        (3.0, 0.9526),
        (-3.0, 0.0474),
        (-1.0, 0.2689),
    ],
)
def test_normalize_rerank_score_logit_mode(raw, expected):
    assert round(normalize_rerank_score(raw, treat_as_logit=True), 4) == expected


def test_normalize_is_monotonic_in_both_modes():
    """两种模式都必须单调 —— 否则阈值就没有意义。

    这条测试是为了钉住一个曾经的错误设计: "逐值判断是不是 logits"
    会在 0 处产生不连续(sigmoid(0)=0.5, 但返回值是 0.0)。
    """
    values = [-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0]
    for mode in (False, True):
        normalized = [normalize_rerank_score(v, treat_as_logit=mode) for v in values]
        assert normalized == sorted(normalized), f"treat_as_logit={mode} 时不单调"
        assert all(0.0 <= v <= 1.0 for v in normalized)


def test_extreme_logits_do_not_overflow():
    assert normalize_rerank_score(1e6, treat_as_logit=True) == pytest.approx(1.0)
    assert normalize_rerank_score(-1e6, treat_as_logit=True) == pytest.approx(0.0)


def test_negative_logits_are_not_mistaken_for_irrelevant():
    """logits 为负不代表"不相关", 只是未归一化。

    这正是原实现用 0.5 卡 logits 会大面积误杀的根因。
    """
    assert normalize_rerank_score(-1.0, treat_as_logit=True) > 0.2
    assert normalize_rerank_score(
        math.log(0.25 / 0.75), treat_as_logit=True
    ) == pytest.approx(0.25, abs=1e-6)


def test_score_semantics_mismatch_is_logged(caplog):
    """配置说"是概率"但实际给出 logits 时, 必须留下 ERROR 日志而不是静默钳制。"""
    node = _node("某条文")
    retriever, _ = _build([NodeWithScore(node=node, score=1.0)])
    retriever._reranker = _FakeReranker({node.text: 4.0})  # 越界 -> 明明不是概率

    with caplog.at_level("ERROR"):
        retriever.retrieve("查询")
    assert any("RERANK_SCORE_IS_LOGIT" in record.message for record in caplog.records)


# ============================================================
# 中文分词
# ============================================================


def test_tokenize_zh_splits_words_not_characters():
    tokens = tokenize_zh("经济补偿按劳动者工作的年限计算")
    assert tokens, "分词结果不能为空"
    # 词级分词的标志: 出现多字词, 而不是所有 token 长度都是 1
    assert any(len(token) > 1 for token in tokens)


def test_tokenize_zh_handles_empty():
    assert tokenize_zh("") == []
    assert tokenize_zh(None) == []


# ============================================================
# T-06 召回与重排分离
# ============================================================


def test_retrieve_raw_does_not_call_reranker():
    node_a = _node("条文甲")
    retriever, index = _build([NodeWithScore(node=node_a, score=0.9)])
    fake = _FakeReranker()
    retriever._reranker = fake

    fused = retriever.retrieve_raw("查询")
    assert len(fused) == 1
    assert fake.calls == 0, "retrieve_raw 不应该触发重排"


def test_rerank_only_runs_once_for_a_query():
    node_a = _node("条文甲")
    retriever, _ = _build([NodeWithScore(node=node_a, score=0.9)])
    fake = _FakeReranker()
    retriever._reranker = fake

    retriever.retrieve("查询")
    assert fake.calls == 1, "单次检索只应重排一次"


def test_rerank_receives_all_candidates():
    nodes = [NodeWithScore(node=_node(f"条文{i}"), score=0.5) for i in range(20)]
    index = _StubIndex(nodes)
    retriever = HybridRetriever(index, nodes=[], use_reranker=True)
    fake = _FakeReranker()
    retriever._reranker = fake

    retriever.retrieve("查询")
    assert fake.last_candidate_count == 20


def test_rrf_fusion_ranks_documents_hit_by_both_paths_first():
    shared = _node("两条路都命中")
    only_vector = _node("只有向量命中")
    only_bm25 = _node("只有BM25命中")

    vector_results = [
        NodeWithScore(node=only_vector, score=0.9),
        NodeWithScore(node=shared, score=0.8),
    ]
    retriever, _ = _build(vector_results, nodes_by_id=[shared, only_bm25])

    fused = retriever.retrieve_raw("两条路都命中")
    assert fused[0].node.node_id == shared.node_id


# ============================================================
# T-02 阈值语义
# ============================================================


def test_threshold_applies_to_normalized_score():
    good = _node("高相关条文")
    bad = _node("低相关条文")
    retriever, _ = _build([NodeWithScore(node=good, score=1.0), NodeWithScore(node=bad, score=1.0)])
    retriever._reranker = _FakeReranker({good.text: 4.0, bad.text: -4.0})

    kept = retriever.retrieve("查询", top_k=5, similarity_threshold=0.5)
    texts = [node.node.text for node in kept]
    assert good.text in texts
    assert bad.text not in texts


def test_threshold_skipped_when_reranker_disabled():
    """关闭重排时 RRF 分数不是相似度, 不能被阈值误杀。"""
    node = _node("某条文")
    index = _StubIndex([NodeWithScore(node=node, score=0.9)])
    retriever = HybridRetriever(index, nodes=[], use_reranker=False)

    kept = retriever.retrieve("查询", top_k=5, similarity_threshold=0.5)
    assert len(kept) == 1, "关闭重排后仍应返回结果, 而不是被阈值全过滤"


def test_empty_query_returns_nothing():
    node = _node("某条文")
    retriever, _ = _build([NodeWithScore(node=node, score=0.9)])
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


# ============================================================
# BM25 语料可重建
# ============================================================


def test_build_bm25_corpus_from_records_sets_ready():
    retriever, _ = _build([])
    assert not retriever.bm25_ready

    count = retriever.set_bm25_corpus_from_records(
        [
            {"id": "n1", "text": "经济补偿按劳动者在本单位工作的年限计算", "metadata": {}},
            {"id": "n2", "text": "用人单位违法解除劳动合同应当支付赔偿金", "metadata": {}},
        ]
    )
    assert count == 2
    assert retriever.bm25_ready
    assert retriever.bm25_size == 2


def test_bm25_path_participates_in_recall_after_corpus_rebuild():
    """重建语料后, BM25 这一路必须真的有召回(对应 T-10 的验收点)。"""
    corpus = [
        _node("经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资"),
        _node("用人单位应当为劳动者缴纳社会保险费"),
    ]
    retriever, _ = _build([], nodes_by_id=[])
    retriever.set_bm25_corpus_from_records(
        [{"id": node.node_id, "text": node.text, "metadata": {}} for node in corpus]
    )
    hits = retriever._bm25_retrieve("经济补偿的年限怎么算", top_k=5)
    assert hits, "BM25 应能召回含'经济补偿'的片段"


# ============================================================
# T-01 引用格式统一
# ============================================================


LAW_META = {
    "law_name": "中华人民共和国劳动合同法",
    "article_no": "第四十七条",
    "chapter_path": "第四章 劳动合同的解除和终止",
    "validity_status": "effective",
    "version_label": "2012年修正",
    "display_text": "第四十七条 经济补偿按劳动者在本单位工作的年限……",
    "filename": "劳动合同法.txt",
}


def test_citation_label_contains_traceable_coordinates():
    label = format_citation_label(LAW_META, 1)
    assert label.startswith("[1]")
    assert "中华人民共和国劳动合同法" in label
    assert "第四十七条" in label
    assert "第四章" in label
    assert "现行有效" in label


def test_context_block_and_sources_use_same_index_numbering():
    nodes = [
        NodeWithScore(node=_node("条文一", **LAW_META), score=0.9),
        NodeWithScore(node=_node("条文二", **{**LAW_META, "article_no": "第四十八条"}), score=0.8),
    ]
    context = build_context_block(nodes)
    sources = build_sources(nodes)

    for index in (1, 2):
        marker = CITATION_INDEX_FORMAT.format(index=index)
        assert marker in context
        assert any(source["index"] == index for source in sources)

    assert [source["index"] for source in sources] == [1, 2]
    assert sources[0]["article_no"] == "第四十七条"
    assert sources[0]["validity_label"] == "现行有效"


def test_sources_carry_node_id_for_traceability():
    nodes = [NodeWithScore(node=_node("条文一", **LAW_META), score=0.9)]
    source = build_sources(nodes)[0]
    assert source["node_id"], "没有 node_id 前端就无法拉取权威原文"
    assert source["text"].startswith("第四十七条")


def test_sources_use_display_text_not_retrieval_prefixed_text():
    """引用展示的必须是干净原文, 不能带上检索用的结构前缀。"""
    prefixed = "《中华人民共和国劳动合同法》 > 第四章 劳动合同的解除和终止 > 第四十七条\n第四十七条 经济补偿……"
    node = _node(prefixed, **LAW_META)
    source = build_sources([NodeWithScore(node=node, score=0.9)])[0]
    assert not source["text"].startswith("《中华人民共和国劳动合同法》 >")
    assert source["text"].startswith("第四十七条")


def test_format_citation_label_degrades_gracefully_for_unknown_metadata():
    label = format_citation_label({}, 3)
    assert label.startswith("[3]")
    assert "未知来源" in label
