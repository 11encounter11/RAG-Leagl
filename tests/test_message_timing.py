# -*- coding: utf-8 -*-
"""耗时落库测试（P2-03，需求 F8.2 缺的那一块）。

`messages` 表此前只有 `created_at`，**没有任何耗时列** —— `response_time` 只在接口
响应里返回过，落库即丢。后果有两条：

1. 需求 §五 写的「每次请求可追溯到…耗时分布」与 §6.4 的 P95 < 3s，**从来没有数据**可验；
2. P2-08「长对话压缩摘要」的验收（20 轮后不线性增长）也拿不到判据。

这里守三件事：
- 耗时真的写进了库；
- 非流式的 `first_token_ms` 必须是 **NULL（不适用）**，不许伪造成 0 ——
  0 会被读成"快得可以忽略"，而真值是"这个量在这条路径上不存在"。
  与 T-31 里"拿不到用量记 None 而不是伪造 0"是同一条原则；
- 流式的首个可见片段确实被计时，且 `first_token_ms <= duration_ms`。

全部脱网可跑：不连 MySQL（写库用桩），不调 LLM。
"""
from __future__ import annotations

import asyncio

from app.services import chat_service as cs
from app.services.chat_service import ChatService


class _FakeMessage:
    def __init__(self, mid):
        self.id = mid


class _FakeConversation:
    id = "conv-1"


class _FakeDb:
    def __init__(self):
        self.calls = []

    def create_conversation(self, title, model_name):
        return _FakeConversation()

    def get_conversation(self, conversation_id):
        return _FakeConversation()

    def get_messages(self, conversation_id, limit=50):
        return []

    def add_message(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeMessage(len(self.calls))


class _FakeRedis:
    async def append_message(self, conversation_id, message):
        return None


def _retrieval(**overrides):
    base = {
        "nodes": ["n1"],
        "context": "参考资料",
        "sources": [{"index": 1, "version_id": "", "article_no": "第十九条"}],
        "workflow_steps": ["retrieval:direct"],
        "reflection_score": None,
        "reflection_note": "",
        "refused": False,
        "refuse_reason": "",
        "domain_score": 0.6,
        "confidence_level": "high",
    }
    base.update(overrides)
    return base


class _StubLLM:
    async def chat_with_usage(self, messages, **kwargs):
        return "试用期最长六个月 [1]", None


class _StubStreamLLM:
    async def chat_stream_with_usage(self, messages, **kwargs):
        yield "试用期"
        yield "最长六个月"


class _StubRouter:
    def __init__(self, llm):
        self._llm = llm

    def get(self, name):
        return self._llm


def _install(monkeypatch, llm):
    fake_db = _FakeDb()
    monkeypatch.setattr(cs, "db_manager", fake_db, raising=True)
    monkeypatch.setattr(cs, "redis_store", _FakeRedis(), raising=True)
    monkeypatch.setattr(cs, "llm_router", _StubRouter(llm), raising=True)
    monkeypatch.setattr(
        ChatService, "_retrieve", lambda self, *a, **k: asyncio.sleep(0, _retrieval())
    )
    return fake_db


def _assistant_row(fake_db):
    return [c for c in fake_db.calls if c.get("role") == "assistant"][0]


# ============================================================
# 非流式
# ============================================================
def test_non_stream_persists_duration_and_no_first_token(monkeypatch):
    """非流式: 有总耗时, 首 token **必须是 None** —— 这条路径没有那个量。"""
    fake_db = _install(monkeypatch, _StubLLM())

    result = asyncio.run(ChatService().chat("试用期最长多久", "deepseek-flash", top_k=5))

    row = _assistant_row(fake_db)
    assert isinstance(row["duration_ms"], int) and row["duration_ms"] >= 0
    assert row["first_token_ms"] is None, (
        "非流式不该伪造 first_token_ms=0, 那会被读成'极快'而不是'不适用'"
    )
    assert result["response_time"] >= 0  # 接口返回值仍然保留, 别把老契约弄断


def test_persist_defaults_leave_both_timing_columns_null(monkeypatch):
    """没给耗时时留 NULL, 而不是 0 —— 让"没测"和"零毫秒"在库里分得开。"""
    fake_db = _FakeDb()
    monkeypatch.setattr(cs, "db_manager", fake_db, raising=True)

    ChatService()._persist("conv-1", "问", "答", "deepseek-flash")

    row = _assistant_row(fake_db)
    assert "duration_ms" in row and row["duration_ms"] is None
    assert "first_token_ms" in row and row["first_token_ms"] is None


# ============================================================
# 流式
# ============================================================
def test_stream_records_first_token_not_later_than_total(monkeypatch):
    fake_db = _install(monkeypatch, _StubStreamLLM())

    async def drain():
        async for _ in ChatService().chat_stream("试用期最长多久", "deepseek-flash", top_k=5):
            pass

    asyncio.run(drain())

    row = _assistant_row(fake_db)
    assert row["first_token_ms"] is not None, "流式必须记到首个可见片段的等待时间"
    assert row["duration_ms"] is not None
    assert row["first_token_ms"] <= row["duration_ms"], "首 token 不可能晚于整个流结束"


def test_refusal_stream_still_records_first_token(monkeypatch):
    """拒答没调 LLM, 但用户确实在等那段固定文案 —— P95 延迟管的是感知, 所以也记。

    这条同时把语义钉住: 别让读数据的人以为 first_token_ms 一定等于"模型首字时间"。
    """
    fake_db = _install(monkeypatch, _StubStreamLLM())
    monkeypatch.setattr(
        ChatService,
        "_retrieve",
        lambda self, *a, **k: asyncio.sleep(
            0, _retrieval(refused=True, refuse_reason="超纲", sources=[], nodes=[], context="")
        ),
    )

    async def drain():
        async for _ in ChatService().chat_stream("商标注册怎么弄", "deepseek-flash", top_k=5):
            pass

    asyncio.run(drain())

    row = _assistant_row(fake_db)
    assert row["first_token_ms"] is not None
    assert row["tokens_used"] == 0, "拒答没调 LLM, 用量 0 是真值"


# ============================================================
# 表结构: 防止有人日后加 default=0
# ============================================================
def test_timing_columns_are_nullable_without_default():
    from app.core.memory.db_store import Message

    for name in ("duration_ms", "first_token_ms"):
        column = Message.__table__.columns[name]
        assert column.nullable is True, f"{name} 必须可空, 否则【没测到】会被写成 0"
        assert column.default is None, f"{name} 不许有默认值 —— 会伪装出零毫秒"
