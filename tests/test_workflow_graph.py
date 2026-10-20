# -*- coding: utf-8 -*-
"""工作流图级测试。

存在的原因: 此前所有工作流测试都直接调用纯函数(assess_domain/classify_domain),
而 langgraph 编译图上发生过一次逃逸 —— 节点签名把 config 标成 Optional[dict],
langgraph 便不注入 RunnableConfig, config.configurable.retriever 恒为 None,
真实请求 100% 报"工作流未注入 retriever", 但 111 个离线测试全绿。
这条测试从编译图入口 ainvoke 跑一遍, 专抓"只有跑起来才暴露"的接线错误。
"""
import asyncio

from llama_index.core.schema import NodeWithScore, TextNode

from app.config.settings import settings
import app.core.workflow.rag_workflow as rw
from app.core.workflow.rag_workflow import create_rag_workflow, params_for_round


def _node(node_id, text, score):
    """真节点类型, 不用 SimpleNamespace。

    2026-10-19 改的: 跨查询 RRF 融合会重建 NodeWithScore(node=TextNode), pydantic 要校验
    node 类型 —— 原来这个桩用 SimpleNamespace 冒充节点, 比真实形状宽松, 于是"融合环节"
    在这个文件里永远跑不到。改成真类型后它才能起到图级测试该有的作用。
    """
    return NodeWithScore(node=TextNode(id_=node_id, text=text), score=score)


class FakeRetriever:
    def __init__(self):
        self.calls = []

    @property
    def rerank_effective(self):
        """本桩的 `rerank()` 返回的是归一化相关度分(0.95/0.70), 所以阈值该生效。"""
        return True

    def retrieve_raw(self, query, law_name=None):
        self.calls.append(("retrieve_raw", query))
        return [_node("a", "第二十一条 试用期最长不得超过六个月。", 0.9),
                _node("b", "第十九条 劳动合同期限三个月以上。", 0.6)]

    def vector_top_score(self, query):
        return 0.35  # high 档: >= DOMAIN_SCORE_CAUTION(0.28)

    def rerank(self, query, nodes, top_n=5):
        self.calls.append(("rerank", query))
        return [
            _node("a", nodes[0].node.text, 0.95),
            _node("b", nodes[1].node.text, 0.70),
        ][:top_n]

    def get_context_text(self, nodes):
        return "\n".join(n.node.text for n in nodes)

    def get_sources_info(self, nodes):
        return [{"index": i + 1, "node_id": n.node.node_id} for i, n in enumerate(nodes)]


class _FakeLLM:
    async def chat(self, messages, **kw):
        return '{"analysis": "片段已足够回答", "confidence": 0.9}'


class _FakeRouter:
    def get(self, name):
        return _FakeLLM()


def _initial_state(query="试用期最长多久"):
    return {
        "query": query,
        "rewritten_query": "",
        "hyde_doc": "",
        "sub_questions": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "context": "",
        "reflection_score": 0.0,
        "reflection_note": "",
        "sources": [],
        "retry_count": 0,
        "retry_params": params_for_round(0),
        "refuse": False,
        "domain_score": 1.0,
        "confidence_level": "high",
        "refuse_reason": "",
        "workflow_steps": [],
    }


def test_compiled_graph_injects_retriever_and_produces_sources(monkeypatch):
    monkeypatch.setattr(rw, "llm_router", _FakeRouter())
    monkeypatch.setattr(settings, "ENABLE_QUERY_REWRITE", False)
    monkeypatch.setattr(settings, "ENABLE_HYDE", False)
    monkeypatch.setattr(settings, "ENABLE_DOMAIN_GATE", True)
    monkeypatch.setattr(settings, "ENABLE_RERANK", True)

    fake = FakeRetriever()
    workflow = create_rag_workflow()
    result = asyncio.run(
        workflow.ainvoke(_initial_state(), config={"configurable": {"retriever": fake}})
    )

    assert not result["refuse"], f"不该拒答, 理由: {result.get('refuse_reason')}"
    assert result["sources"], "引用列表不能为空(retriever 注入失败时这里必空)"
    assert result["context"]
    assert "generation_ready" in result["workflow_steps"]
    # 纪律性断言: 召回只按查询路数调用, 重排整图只发生一次(T-06)
    assert sum(1 for c in fake.calls if c[0] == "rerank") == 1
