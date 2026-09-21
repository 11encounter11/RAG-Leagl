"""
拒答闸门测试

守的是一条安全底线: 问题超出知识库范围时, 必须拒答, 而不是拿手边的条文硬答。
这条线以前挂在重排器上, 而重排器是可选组件 —— 实测中它加载不了,
结果 4 条超纲题全部返回了劳动法条文(拒答准确率 0%), 系统没有任何异常表现。

所以这里测的是"独立性": 判据必须只依赖向量相似度, 不依赖重排器。
"""
from __future__ import annotations

import pytest

from app.config.settings import settings
from app.core.rag.retriever import HybridRetriever, NodeWithScore, TextNode
from app.core.workflow.rag_workflow import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_OUT_OF_DOMAIN,
    classify_domain,
)


# ============================================================
# 档位映射
# ============================================================
def test_below_reject_line_is_out_of_domain():
    assert (
        classify_domain(settings.DOMAIN_SCORE_REJECT - 0.01) == CONFIDENCE_OUT_OF_DOMAIN
    )


def test_between_lines_is_low_confidence():
    mid = (settings.DOMAIN_SCORE_REJECT + settings.DOMAIN_SCORE_CAUTION) / 2
    assert classify_domain(mid) == CONFIDENCE_LOW


def test_above_caution_line_is_high_confidence():
    assert classify_domain(settings.DOMAIN_SCORE_CAUTION + 0.01) == CONFIDENCE_HIGH


def test_boundary_values_are_not_treated_as_out_of_domain():
    """恰好等于拒答线时不应拒答 —— 边界必须偏保守, 避免把正常问题误杀。"""
    assert classify_domain(settings.DOMAIN_SCORE_REJECT) == CONFIDENCE_LOW
    assert classify_domain(settings.DOMAIN_SCORE_CAUTION) == CONFIDENCE_HIGH


def test_thresholds_are_ordered():
    assert settings.DOMAIN_SCORE_REJECT < settings.DOMAIN_SCORE_CAUTION


def test_gate_can_be_disabled():
    """开关关掉后应一律放行 —— 否则"关掉开关"就是伪生效。"""
    original = settings.ENABLE_DOMAIN_GATE
    try:
        settings.ENABLE_DOMAIN_GATE = False
        assert classify_domain(0.0) == CONFIDENCE_HIGH
        assert classify_domain(-1.0) == CONFIDENCE_HIGH
    finally:
        settings.ENABLE_DOMAIN_GATE = original


# ============================================================
# 判据来源: 只依赖向量, 不依赖重排器
# ============================================================
class _StubVectorRetriever:
    def __init__(self, nodes, error=None):
        self._nodes = nodes
        self._error = error

    def retrieve(self, query):
        if self._error:
            raise self._error
        return list(self._nodes)


class _StubIndex:
    def __init__(self, nodes, error=None):
        self._nodes = nodes
        self._error = error

    def as_retriever(self, similarity_top_k=None):
        return _StubVectorRetriever(self._nodes, self._error)


def _make_retriever(vector_nodes, vector_error=None):
    return HybridRetriever(_StubIndex(vector_nodes, vector_error), nodes=[], use_reranker=False)


def test_vector_top_score_returns_top_similarity():
    nodes = [
        NodeWithScore(node=TextNode(text="a"), score=0.42),
        NodeWithScore(node=TextNode(text="b"), score=0.31),
    ]
    assert _make_retriever(nodes).vector_top_score("随便问") == pytest.approx(0.42)


def test_vector_top_score_zero_when_nothing_retrieved():
    """检索为空说明连最近的片段都没有 —— 这本身就是"不在本知识库范围"的强信号。"""
    assert _make_retriever([]).vector_top_score("商标怎么注册") == 0.0


def test_vector_top_score_works_without_reranker():
    """核心: 判据必须在不重排的情况下可用。

    这是本次修复的要点 —— 原实现的拒答依赖重排分数, 重排器不可用时
    拒答闸门直接失效。
    """
    nodes = [NodeWithScore(node=TextNode(text="a"), score=0.15)]
    retriever = _make_retriever(nodes)
    assert retriever.reranker_ready is False
    assert retriever.vector_top_score("超纲问题") == pytest.approx(0.15)
    # 而且这个分数确实会被判成拒答
    assert classify_domain(retriever.vector_top_score("超纲问题")) == CONFIDENCE_OUT_OF_DOMAIN


def test_vector_failure_does_not_block_the_request():
    """判据本身出错时宁可放行, 也不能让整个问答请求挂掉。"""
    retriever = _make_retriever([], vector_error=RuntimeError("Milvus 挂了"))
    score = retriever.vector_top_score("任意问题")
    assert score == 1.0, "拿不到信号应返回满分放行, 而不是 0.0(那会把所有问题都拒掉)"
    assert classify_domain(score) == CONFIDENCE_HIGH
