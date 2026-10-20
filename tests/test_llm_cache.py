# -*- coding: utf-8 -*-
"""查询改写/拆分的结果缓存与留痕测试（2026-10-19）。

为什么必须有：改写句决定"拿哪句话去检索"，而实测同一句问题在 `temperature=0` 下
**三次改写得到三个版本**（有的丢了"社保征收机构"这个关键词）。输入每次都变，
agentic 路径的评测就没有可比性——门禁"不低于基线"根本没法判。

缓存顺带解决第二件事：以前 `rewritten_query` 连工作流的返回值都不在，
线上出问题时只能自己包一层 LLM 才看得见它到底写成了什么。现在它进留痕。

全部脱网可跑：Redis 用内存假件，LLM 用计数器假件。
"""
from __future__ import annotations

import asyncio

from llama_index.core.schema import NodeWithScore, TextNode

import app.core.workflow.rag_workflow as rw
from app.config.settings import settings
from app.core.workflow.rag_workflow import create_rag_workflow, params_for_round

REWRITE_MARK = "改写成更适合在法律"


def _node(nid):
    return NodeWithScore(
        node=TextNode(id_=nid, text=f"第{nid}条 内容", metadata={"validity_status": "effective"}),
        score=0.5,
    )


class FakeRetriever:
    rerank_effective = False

    def retrieve_raw(self, query, law_name=None):
        return [_node(f"n{i}") for i in range(6)]

    def vector_top_score(self, query):
        return 0.40

    def rerank(self, query, nodes, top_n=5):
        return nodes[:top_n]

    def get_context_text(self, nodes):
        return "\n".join(n.node.text for n in nodes)

    def get_sources_info(self, nodes):
        return [{"index": i + 1, "node_id": n.node.node_id} for i, n in enumerate(nodes)]


class FakeRedis:
    """内存假件；`broken=True` 时模拟 Redis 挂掉。"""

    def __init__(self, broken=False):
        self.store = {}
        self.broken = broken
        self.reads = 0

    async def cache_get(self, key):
        self.reads += 1
        if self.broken:
            raise ConnectionError("Redis 未启动")
        return self.store.get(key)

    async def cache_set(self, key, value, ttl=None):
        if self.broken:
            raise ConnectionError("Redis 未启动")
        self.store[key] = value


class CountingLLM:
    model_name = "stub-rewrite-model"   # 真 LLM 一定有(BaseLLM.__init__ 设的)

    def __init__(self):
        self.rewrite_calls = 0

    async def chat(self, messages, **kw):
        prompt = messages[-1]["content"]
        if REWRITE_MARK in prompt:
            self.rewrite_calls += 1
            return "用人单位未依法缴纳社会保险费 补缴 维权途径"
        return '{"analysis": "足够", "confidence": 0.9}'


class Router:
    def __init__(self, llm):
        self._llm = llm

    def get(self, name):
        return self._llm


def _state(q):
    return {
        "query": q, "rewritten_query": "", "hyde_doc": "", "sub_questions": [],
        "retrieved_docs": [], "reranked_docs": [], "context": "",
        "reflection_score": 0.0, "reflection_note": "", "sources": [],
        "retry_count": 0, "retry_params": params_for_round(0),
        "refuse": False, "domain_score": 1.0, "confidence_level": "high",
        "refuse_reason": "", "workflow_steps": [],
    }


def _run(monkeypatch, redis):
    llm = CountingLLM()
    monkeypatch.setattr(rw, "llm_router", Router(llm))
    monkeypatch.setattr(rw, "redis_store", redis)
    monkeypatch.setattr(settings, "ENABLE_QUERY_REWRITE", True)
    monkeypatch.setattr(settings, "ENABLE_HYDE", False)
    monkeypatch.setattr(settings, "ENABLE_DOMAIN_GATE", False)
    monkeypatch.setattr(settings, "REFLECTION_THRESHOLD", 0.5)
    return llm, asyncio.run(
        create_rag_workflow().ainvoke(
            _state("公司一直不给我交社保怎么办"),
            config={"configurable": {"retriever": FakeRetriever()}},
        )
    )


def test_second_identical_question_does_not_pay_for_rewrite_again(monkeypatch):
    redis = FakeRedis()
    llm1, _ = _run(monkeypatch, redis)
    assert llm1.rewrite_calls == 1
    llm2, result2 = _run(monkeypatch, redis)
    assert llm2.rewrite_calls == 0, "同一个问题第二次问, 改写不该再花一次调用"
    assert "query_analysis:cache" in result2["workflow_steps"], "命中缓存要留痕"


def test_different_question_does_not_hit_cache(monkeypatch):
    redis = FakeRedis()
    _run(monkeypatch, redis)
    llm = CountingLLM()
    monkeypatch.setattr(rw, "llm_router", Router(llm))
    monkeypatch.setattr(rw, "redis_store", redis)
    monkeypatch.setattr(settings, "ENABLE_QUERY_REWRITE", True)
    monkeypatch.setattr(settings, "ENABLE_HYDE", False)
    monkeypatch.setattr(settings, "ENABLE_DOMAIN_GATE", False)
    asyncio.run(
        create_rag_workflow().ainvoke(
            _state("试用期最长可以约定多久"),
            config={"configurable": {"retriever": FakeRetriever()}},
        )
    )
    assert llm.rewrite_calls == 1, "换了问题必须重新改写, 不能拿上一题的结果"


def test_redis_being_down_does_not_break_the_answer(monkeypatch):
    """缓存是加速件, 不是单点。Redis 挂掉必须照常回答, 只是回到"每次都问模型"。

    注意要注入**坏客户端**而不是"自己会抛异常的假 store" —— 吞异常的逻辑写在
    RedisStore 内部, 换一个直接抛的假件等于把被测的那段绕开了(第一版就犯了这个错)。
    """
    from app.core.memory.redis_store import RedisStore

    class _BrokenClient:
        async def get(self, key):
            raise ConnectionError("Redis 未启动")

        async def setex(self, key, ttl, value):
            raise ConnectionError("Redis 未启动")

    store = RedisStore()
    store._redis = _BrokenClient()

    llm, result = _run(monkeypatch, store)
    assert llm.rewrite_calls == 1, "缓存不可用时仍要照常改写"
    assert not result["refuse"]
    assert result["reranked_docs"], "Redis 故障不该把候选清空"


def test_empty_rewrite_is_not_cached(monkeypatch):
    """空输出写进缓存会把后续所有请求都锁死在空值上。"""
    redis = FakeRedis()
    calls = {"n": 0}

    class EmptyLLM:
        model_name = "stub-rewrite-model"

        async def chat(self, messages, **kw):
            # 只数改写那一次: 反思和生成也走同一个 llm, 全记下来断言就错了
            if REWRITE_MARK in messages[-1]["content"]:
                calls["n"] += 1
                return "   "
            return '{"analysis": "足够", "confidence": 0.9}'

    monkeypatch.setattr(rw, "llm_router", Router(EmptyLLM()))
    monkeypatch.setattr(rw, "redis_store", redis)
    monkeypatch.setattr(settings, "ENABLE_QUERY_REWRITE", True)
    monkeypatch.setattr(settings, "ENABLE_HYDE", False)
    monkeypatch.setattr(settings, "ENABLE_DOMAIN_GATE", False)
    for _ in range(2):
        asyncio.run(
            create_rag_workflow().ainvoke(
                _state("社保没交怎么办"),
                config={"configurable": {"retriever": FakeRetriever()}},
            )
        )
    assert redis.store == {}, "空字符串不该被缓存"
    assert calls["n"] == 2, "缓存没存住才对, 两次都该重新问"


def test_self_created_client_is_rebuilt_when_the_loop_changes():
    """缓存要跨 `asyncio.run` 活得下来 —— 评测脚本每题一个新循环。

    redis.asyncio 的连接池绑在创建它的循环上, 换循环再用直接抛
    `Event loop is closed`; 而 cache_get 把异常当"未命中", 于是**静默地永远不命中**。

    这条不连真的 Redis: `from_url` 是懒连接, 没发命令就不建 socket, 所以脱网也能跑。
    """
    from app.core.memory.redis_store import RedisStore

    store = RedisStore()

    async def _first():
        return await store.get_client()

    client_a = asyncio.run(_first())
    assert store._loop is not None, "自建的客户端要记下所属循环, 才知道什么时候该换"

    async def _second():
        return await store.get_client()

    client_b = asyncio.run(_second())
    assert client_b is not client_a, "换了循环必须重建客户端, 而不是拿旧循环那个"


def test_injected_client_is_never_replaced(monkeypatch):
    """重建只管自己创建的那个。外部塞进来的假件一律照用。

    这条是踩过之后补的: 早先的重建条件不看来源, 于是"注入坏客户端测降级"的测试
    拿着真连接去读本机 Redis —— 第一次跑绿, 第二次因为上一轮把值写进去了就红。
    """
    from app.core.memory.redis_store import RedisStore

    class _Injected:
        async def get(self, key):  # pragma: no cover - 断言用不到它被调用
            return None

    store = RedisStore()
    injected = _Injected()
    store._redis = injected

    async def _check():
        return await store.get_client()

    assert asyncio.run(_check()) is injected
    assert store._loop is None


def test_trace_records_the_rewritten_query():
    """留痕里必须能看到"当时到底用哪句话去检索的"。"""
    from app.services.chat_service import build_retrieval_info

    info = build_retrieval_info(
        {"sources": [], "rewritten_query": "用人单位未依法缴纳社会保险费 补缴"},
        model_name="deepseek-flash", top_k=5,
    )
    assert info["rewritten_query"] == "用人单位未依法缴纳社会保险费 补缴"

    direct = build_retrieval_info({"sources": []}, model_name="m", top_k=5)
    assert direct["rewritten_query"] == "", "字段缺失时给空串, 不要让留痕抛异常"
