# -*- coding: utf-8 -*-
"""token 计量测试（T-31 的后半截，需求 F8.3）。

`messages.tokens_used` 此前恒为 0。根因不在落库那层，而在适配器：
`deepseek.py` 只 `return response.choices[0].message.content`，接口其实回了
`usage`，但被扔掉了 —— 所以下游**根本拿不到数**，落库怎么写都是 0。

这里守四件事：
1. 用量真的从接口取回来并传到 `add_message`；
2. **重生成时两次调用要累加** —— 只记最后一次会把成本读低近一半；
3. 流式的用量对象**不能混进正文**推给前端；
4. 没实现用量的适配器退化成 `None`（落库 0），而不是报错。

全部脱网可跑：LLM 客户端用桩替换。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.core.llm.base import BaseLLM, LLMUsage
from app.core.llm.deepseek import DeepSeekFlashLLM
from app.services import chat_service as cs
from app.services.chat_service import ChatService, _tokens_of


class _StubLLM:
    def __init__(self, replies, usages=None):
        self.replies = list(replies)
        self.usages = list(usages) if usages else [LLMUsage(100, 20, 120)] * len(replies)
        self.calls = []

    async def chat_with_usage(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0), self.usages.pop(0)


def _retrieval():
    return {"sources": [{"index": 1, "version_id": "", "article_no": "第四十七条"}]}


# ============================================================
# 1) 累加，而不是只记最后一次
# ============================================================
def test_single_generation_records_its_tokens():
    llm = _StubLLM(["根据《劳动合同法》第四十七条 [1] 支付。"])
    answer, spent = asyncio.run(
        ChatService()._generate_checked(llm, [{"role": "user", "content": "x"}], _retrieval())
    )
    assert answer.startswith("根据")
    assert spent == 120


def test_regeneration_accumulates_both_calls():
    """重生成是又烧了一整份长上下文。只记最后一次 = 把成本读低一半。"""
    llm = _StubLLM(
        ["依据 [7] 处理。", "根据《劳动合同法》第四十七条 [1] 支付。"],
        usages=[LLMUsage(100, 20, 120), LLMUsage(110, 25, 135)],
    )
    answer, spent = asyncio.run(
        ChatService()._generate_checked(llm, [{"role": "user", "content": "x"}], _retrieval())
    )
    assert len(llm.calls) == 2
    assert spent == 255, "两次调用的用量必须相加"


def test_refusal_costs_zero():
    """拒答那次根本没调 LLM —— 0 是真值，不是"没测到"。"""
    assert _tokens_of(None) == 0


# ============================================================
# 2) 没有用量时的退化
# ============================================================
def test_base_default_chat_with_usage_wraps_chat():
    """没覆写 `chat_with_usage` 的子类, 必须退化成 (chat 的结果, None)。

    这条是给 GLM 那种"保留但没注册"的实现兜底的：不能因为没实现用量就整条链路报错。
    """

    class _Legacy(BaseLLM):
        async def chat(self, messages, **kwargs):
            return "老实现"

        async def chat_stream(self, messages, **kwargs):
            yield "块"

    text, usage = asyncio.run(_Legacy("m", "k").chat_with_usage([]))
    assert text == "老实现"
    assert usage is None, "拿不到用量时必须是 None, 不能伪造 0"


# ============================================================
# 3) 传到落库
# ============================================================
class _FakeMessage:
    id = 7


class _FakeDb:
    def __init__(self):
        self.calls = []

    def add_message(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeMessage()


def test_persist_passes_tokens_used_to_add_message(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(cs, "db_manager", fake, raising=True)

    ChatService()._persist("c1", "问", "答", "deepseek-flash", tokens_used=388)

    assert fake.calls[0]["role"] == "user"
    assert "tokens_used" not in fake.calls[0], "用户消息没有用量, 不要塞个 0 冒充测过"
    assert fake.calls[1]["tokens_used"] == 388


# ============================================================
# 4) DeepSeek 适配器: 用量真的从接口取回来了
# ============================================================
def _fake_client(monkeypatch, llm, create_impl):
    monkeypatch.setattr(llm.client.chat.completions, "create", create_impl, raising=True)


async def _collect(agen):
    return [item async for item in agen]


def test_deepseek_non_stream_returns_usage(monkeypatch):
    llm = DeepSeekFlashLLM(api_key="k")

    async def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="答案文本"))],
            usage=SimpleNamespace(prompt_tokens=77, completion_tokens=23, total_tokens=100),
        )

    _fake_client(monkeypatch, llm, create)
    text, usage = asyncio.run(llm.chat_with_usage([{"role": "user", "content": "hi"}]))

    assert text == "答案文本"
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (77, 23, 100)
    # 旧行为必须保持不变: chat() 仍然只给文字
    assert asyncio.run(llm.chat([{"role": "user", "content": "hi"}])) == "答案文本"


def test_deepseek_stream_requests_usage_and_keeps_it_out_of_the_body(monkeypatch):
    """两个断言合一条:

    ① 必须显式带 `stream_options={"include_usage": true}` —— 不带就拿不到用量,
       而且**失效是静默的**（表现只是 tokens_used 一直是 0）;
    ② 末尾那个用量对象不能混进正文 —— 它会被当文字推给前端。
    """
    llm = DeepSeekFlashLLM(api_key="k")
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)

        async def gen():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="一段"))], usage=None
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="二段"))], usage=None
            )
            # 带 include_usage 时, 最后一个 chunk 的 choices 为空、只带 usage
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=9, completion_tokens=6, total_tokens=15),
            )

        return gen()

    _fake_client(monkeypatch, llm, create)

    items = asyncio.run(_collect(llm.chat_stream_with_usage([{"role": "user", "content": "x"}])))
    assert seen.get("stream_options") == {"include_usage": True}
    assert items[:2] == ["一段", "二段"]
    assert isinstance(items[-1], LLMUsage) and items[-1].total_tokens == 15

    # 老的 chat_stream 契约不能变: 只有文字
    plain = asyncio.run(_collect(llm.chat_stream([{"role": "user", "content": "x"}])))
    assert plain == ["一段", "二段"]
