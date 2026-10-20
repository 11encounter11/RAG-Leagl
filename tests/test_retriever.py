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
        # 记录最近一次 as_retriever 收到的过滤条件 —— 用来断言"过滤确实下推了"
        self.last_filters = "NOT_CALLED"

    def as_retriever(self, similarity_top_k=None, filters=None):
        self.last_filters = filters
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
    # 真实 chunk 一定带 validity_status（元数据契约的必填字段，见 metadata.py）。
    # 这里给个真实默认值，避免测试桩造出"现实中不存在"的节点 ——
    # 时效过滤开启时，缺这个字段的片段会被判为不合格而排除（fail-closed）。
    metadata.setdefault("validity_status", "effective")
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


def test_threshold_gate_uses_top1_and_keeps_lower_ranks():
    """阈值只决定"这一批用不用", 不逐条砍 —— 第 2~5 名不因绝对分低被丢。

    旧行为是逐条 `score >= threshold`。2026-10-19 实测 L40 的期望条文重排后
    排第 3、rel 只有 0.0140, 被 0.1 砍掉后整批 5 条只剩 1 条喂给模型(P2-20)。
    """
    good = _node("高相关条文")
    low_but_ranked = _node("排第 2 但绝对分低")
    retriever, _ = _build(
        [NodeWithScore(node=good, score=1.0), NodeWithScore(node=low_but_ranked, score=1.0)]
    )
    retriever._reranker = _FakeReranker({good.text: 0.8, low_but_ranked.text: -4.0})

    kept = retriever.retrieve("查询", top_k=5, similarity_threshold=0.5)

    assert [node.node.text for node in kept] == [good.text, low_but_ranked.text], (
        "top1 达标后整批保留, 不再逐条砍"
    )
    info = retriever.last_threshold_info
    assert info["scale"] == "rerank_relevance"
    assert info["rule"] == "top1_gate"
    assert info["value"] == 0.5
    assert info["applied"] is True
    assert info["top1_score"] > 0.5
    assert info["returned"] == 2


def test_rerank_on_returns_top_k_when_top1_passes():
    """P2-20 验收③: 开重排后, 只要 top1 达标, 返回条数不得少于 top_k。

    候选比 top_k 多, 且第 2 名之后全部远低于阈值 —— 旧实现只会返回 1 条。
    """
    nodes = [NodeWithScore(node=_node(f"条文{i}"), score=0.5) for i in range(6)]
    retriever, _ = _build(nodes)
    retriever._reranker = _FakeReranker(
        {nodes[0].node.text: 0.8, **{n.node.text: -4.0 for n in nodes[1:]}}
    )

    kept = retriever.retrieve("查询", top_k=5, similarity_threshold=0.5)

    assert len(kept) == 5, f"top1 达标却只返回 {len(kept)} 条(阈值又吃坑位了)"
    assert kept[0].node.text == nodes[0].node.text

    rejected = retriever.retrieve("查询", top_k=5, similarity_threshold=0.9)
    assert rejected == [], "top1 自己都不达标时才整批不用"
    assert retriever.last_threshold_info["returned"] == 0


def test_threshold_skipped_when_reranker_disabled():
    """关闭重排时 RRF 分数不是相似度, 不能被阈值误杀。"""
    node = _node("某条文")
    index = _StubIndex([NodeWithScore(node=node, score=0.9)])
    retriever = HybridRetriever(index, nodes=[], use_reranker=False)

    kept = retriever.retrieve("查询", top_k=5, similarity_threshold=0.5)
    assert len(kept) == 1, "关闭重排后仍应返回结果, 而不是被阈值全过滤"
    assert retriever.last_threshold_info["applied"] is False
    assert retriever.last_threshold_info["scale"] == "rrf_rank_scores"


def test_threshold_skipped_when_reranker_fails_to_load():
    """回归: 重排"配置开着"但**加载失败**时, 同样不能套相似度阈值。

    这是真实踩过的坑(2026-10-14): 开发机内存不足, CrossEncoder 加载抛
    OSError(页面文件太小), 于是分数退回 RRF 原值(0.03 量级), 而阈值 0.1
    把它们全部误杀 —— 表现是"评测 52 题全部召回为空、系统什么都查不到",
    日志里只有一行 warning。

    与 `use_reranker=False` 是同一个陷阱的两个入口, 之前只堵了后者。
    """
    node = _node("某条文")
    index = _StubIndex([NodeWithScore(node=node, score=0.03)])
    retriever = HybridRetriever(index, nodes=[], use_reranker=True)

    # 模拟"加载失败": _load_reranker 之后 self._reranker 仍为 None
    retriever._load_reranker = lambda: None
    retriever._reranker = None

    kept = retriever.retrieve("查询", top_k=5, similarity_threshold=0.1)

    assert len(kept) == 1, "重排器不可用时不应把 RRF 分数当相似度去卡阈值"
    assert retriever._last_rerank_ok is False


def test_threshold_applied_only_when_rerank_really_ran():
    """对照组: 重排真的执行过时, 阈值必须照常生效(别把修复做成"永远不过阈值")。"""
    node = _node("低分条文")
    index = _StubIndex([NodeWithScore(node=node, score=0.03)])
    retriever = HybridRetriever(index, nodes=[], use_reranker=True)
    retriever._reranker = _FakeReranker({node.text: -4.0})  # 归一化后接近 0

    kept = retriever.retrieve("查询", top_k=5, similarity_threshold=0.5)

    assert retriever._last_rerank_ok is True
    assert kept == [], "重排生效时阈值仍应把低分候选过滤掉"


def test_empty_query_returns_nothing():
    node = _node("某条文")
    retriever, _ = _build([NodeWithScore(node=node, score=0.9)])
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


# ============================================================
# 时效过滤（T-27 / T-28）
#
# 语料里同一部法有两个版本（劳动法 2009/2018、社保法 2010/2018），同一条条文会有
# 两条记录。过滤必须在**取 top-k 之前**生效，否则坑位被旧版本占掉、有效版本被挤出。
# ============================================================
def _eff(text: str) -> TextNode:
    return TextNode(text=text, metadata={"validity_status": "effective", "article_no": text})


def _amd(text: str) -> TextNode:
    return TextNode(text=text, metadata={"validity_status": "amended", "article_no": text})


def test_filter_conditions_follow_the_switch(monkeypatch):
    from app.config.settings import settings

    retriever, _ = _build([NodeWithScore(node=_eff("x"), score=0.9)])

    monkeypatch.setattr(settings, "ENABLE_VALIDITY_FILTER", True)
    assert retriever._filter_conditions() == [("validity_status", "effective")]

    monkeypatch.setattr(settings, "ENABLE_VALIDITY_FILTER", False)
    assert retriever._filter_conditions() == []


def test_metadata_filters_are_pushed_to_milvus(monkeypatch):
    """向量那一路：过滤应作为 filters 传给 as_retriever（→ Milvus expr 下推）。"""
    from app.config.settings import settings

    retriever, index = _build([NodeWithScore(node=_eff("x"), score=0.9)])

    monkeypatch.setattr(settings, "ENABLE_VALIDITY_FILTER", True)
    retriever.retrieve_raw("查询")
    assert index.last_filters is not None, "开启时效过滤时，必须把 filters 传给 Milvus"
    assert index.last_filters.filters[0].key == "validity_status"
    assert index.last_filters.filters[0].value == "effective"

    monkeypatch.setattr(settings, "ENABLE_VALIDITY_FILTER", False)
    retriever.retrieve_raw("查询")
    assert index.last_filters is None, "关闭时效过滤时不应传 filters"


def test_passes_filter_predicate(monkeypatch):
    from app.config.settings import settings

    retriever, _ = _build([NodeWithScore(node=_eff("x"), score=0.9)])

    monkeypatch.setattr(settings, "ENABLE_VALIDITY_FILTER", True)
    assert retriever.passes_filter({"validity_status": "effective"}) is True
    assert retriever.passes_filter({"validity_status": "amended"}) is False
    assert retriever.passes_filter({}) is False, "缺字段不该被当成合格"

    monkeypatch.setattr(settings, "ENABLE_VALIDITY_FILTER", False)
    assert retriever.passes_filter({"validity_status": "amended"}) is True


def test_bm25_path_excludes_non_effective_versions(monkeypatch):
    """BM25 是内存索引，没有查询层可下推 —— 必须在取 top-k 之前筛。"""
    from app.config.settings import settings

    nodes = [_amd("旧版条文 甲乙丙"), _eff("现行条文 甲乙丙")]
    retriever = HybridRetriever(_StubIndex([]), nodes=nodes, use_reranker=False)

    monkeypatch.setattr(settings, "ENABLE_VALIDITY_FILTER", True)
    kept = retriever._bm25_retrieve("甲乙丙", top_k=5)
    assert [n.node.text for n in kept] == ["现行条文 甲乙丙"]

    monkeypatch.setattr(settings, "ENABLE_VALIDITY_FILTER", False)
    both = retriever._bm25_retrieve("甲乙丙", top_k=5)
    assert len(both) == 2, "关掉开关后旧版本应能召回"


def test_bm25_filter_happens_before_top_k(monkeypatch):
    """关键：过滤必须在取 top-k **之前**。

    构造 3 条语料，让**被过滤掉的那两条**打分更高。若实现是"先取 top-2 再筛"，
    结果会是空的；正确实现应返回第 3 条（唯一合格的那条）。
    """
    from app.config.settings import settings

    # 旧版本条文重复更多次关键词 → BM25 分更高，但会被过滤掉
    nodes = [
        _amd("甲乙丙 甲乙丙 甲乙丙"),
        _amd("甲乙丙 甲乙丙"),
        _eff("甲乙丙"),
    ]
    retriever = HybridRetriever(_StubIndex([]), nodes=nodes, use_reranker=False)
    monkeypatch.setattr(settings, "ENABLE_VALIDITY_FILTER", True)

    kept = retriever._bm25_retrieve("甲乙丙", top_k=2)
    assert [n.node.metadata["validity_status"] for n in kept] == ["effective"], (
        "top-2 若被旧版本占掉，说明过滤发生在 top-k 之后"
    )


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
        [
            # metadata 要带 validity_status —— 真实重建链路从 Milvus 读回的元数据里有它
            {"id": node.node_id, "text": node.text, "metadata": dict(node.metadata)}
            for node in corpus
        ]
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


# ============================================================
# D-2: 重排模型可配置化(RERANK_TORCH_DTYPE / RERANK_BACKEND)
# ============================================================


def test_cross_encoder_reranker_rejects_unknown_dtype_before_loading_model():
    """非法 dtype 必须在真正加载 1GB 模型之前就报错。

    否则错误只会在 CrossEncoder 内部深处抛出, 排查成本高; 而且如果先下载/
    加载再报错, 在内存吃紧的机器上(N-9)这一步本身就可能把机器打挂。
    """
    from app.core.rag.retriever import _CrossEncoderReranker

    with pytest.raises(ValueError, match="RERANK_TORCH_DTYPE"):
        _CrossEncoderReranker(
            model="BAAI/bge-reranker-base",
            top_n=5,
            torch_dtype="int8",  # 不在 float32/float16/bfloat16 之列
        )


def test_cross_encoder_reranker_dtype_validation_skipped_for_non_torch_backend(monkeypatch):
    """onnx/openvino 后端不走 torch 的 dtype 分支, 不应该被 torch_dtype 校验拦住。

    刻意用 monkeypatch 替换掉真正的 CrossEncoder, 而不是传一个"不存在的模型名"
    去让它自然报错 —— 后者会触发 sentence-transformers 去 huggingface.co 发 HEAD
    请求核对文件, 网络不通时会重试若干次(每次退避翻倍), 单元测试不该有这种
    几十秒的卡顿, 更不该依赖网络。
    """
    import sentence_transformers

    from app.core.rag.retriever import _CrossEncoderReranker

    captured_kwargs = {}

    class _StubCrossEncoder:
        def __init__(self, model_name_or_path, **kwargs):
            captured_kwargs["model"] = model_name_or_path
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _StubCrossEncoder)

    reranker = _CrossEncoderReranker(
        model="any/model-name",
        top_n=5,
        backend="onnx",
        torch_dtype="int8",  # 若 backend!=torch 时仍校验 dtype, 这里会先抛 ValueError
    )
    assert reranker.backend == "onnx"
    assert captured_kwargs["backend"] == "onnx"
    # backend != torch 时不应该往 model_kwargs 里塞 torch_dtype
    assert captured_kwargs.get("model_kwargs") is None
