# -*- coding: utf-8 -*-
"""引用校验的降级路径测试（T-30，需求 F2.8 / 设计 02 §6.6）。

守的是"引用对不上时会发生什么"：
    生成 → 校验 → 有问题就**带反馈重生成一次** → 仍有问题则标注"部分依据未能核验"。

为什么必须测这条：降级路径平时几乎不触发（正常语料下引用都能核验），
所以它是典型的"写错了也没人发现"的代码 —— 真出事那天才发现重试根本没生效。

全部脱网可跑：LLM 用桩替换，查权威表的函数按用例注入/打桩。
"""
from __future__ import annotations

import asyncio

import pytest

from app.config.settings import settings
from app.core.llm.base import LLMUsage
from app.core.memory.db_store import db_manager
from app.services.chat_service import (
    CITATION_REPAIR_HINT,
    UNVERIFIED_BADGE,
    chat_service,
)


class _StubLLM:
    """按顺序吐出预设回答的假 LLM；记录下来收到的消息，便于断言"反馈有没有带上"。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, messages):
        answer, _ = await self.chat_with_usage(messages)
        return answer

    async def chat_with_usage(self, messages):
        """T-31 之后服务层调的是这个。用量给个可辨识的固定值, 便于断言"有没有累加"。"""
        self.calls.append(messages)
        return self.replies.pop(0), LLMUsage(100, 20, 120)


def _retrieval(**overrides) -> dict:
    source = {
        "index": 1,
        "version_id": "",
        "article_no": "第四十七条",
        "section_seq": 46,
    }
    source.update(overrides)
    return {"sources": [source]}


def _generate(llm, retrieval, message="被辞退了有补偿吗"):
    """只取答案文本 —— 用量那一路的断言在 tests/test_token_usage.py。"""
    answer, _tokens_used = asyncio.run(
        chat_service._generate_checked(llm, [{"role": "user", "content": message}], retrieval)
    )
    return answer


# ============================================================
# 通过时不重生成
# ============================================================
def test_clean_answer_is_returned_without_regeneration():
    llm = _StubLLM(["根据《劳动合同法》第四十七条 [1]，应按工作年限支付经济补偿。"])
    answer = _generate(llm, _retrieval())
    assert answer.startswith("根据")
    assert len(llm.calls) == 1, "引用没问题却重生成了一次, 白花一次调用"


def test_answer_without_any_citation_is_untouched():
    """没有引用的回答不该被改 —— 追问、拒答都不带引用。"""
    llm = _StubLLM(["现有资料中未找到相关依据。"])
    assert _generate(llm, _retrieval()) == "现有资料中未找到相关依据。"
    assert len(llm.calls) == 1


# ============================================================
# 有问题 -> 重生成一次
# ============================================================
def test_bad_citation_triggers_one_regeneration():
    llm = _StubLLM(
        ["依据 [7] 处理。", "根据《劳动合同法》第四十七条 [1]，应按工作年限支付经济补偿。"]
    )
    answer = _generate(llm, _retrieval())

    assert len(llm.calls) == 2, "引用越界没有触发重生成"
    assert answer.startswith("根据")
    assert UNVERIFIED_BADGE.split("{")[0] not in answer, "修好了就不该再挂降级标注"


def test_regeneration_hint_names_the_offending_citation():
    """重生成必须**说清楚哪条有问题** —— 笼统说'再检查一遍'等于原样重试。"""
    llm = _StubLLM(["依据 [7] 处理。", "根据 [1] 处理。"])
    _generate(llm, _retrieval())

    repair_message = llm.calls[1][-1]["content"]
    assert repair_message == CITATION_REPAIR_HINT.format(citations="[7]")
    assert "[7]" in repair_message


def test_regeneration_only_happens_once():
    """两次都不行就停手 —— 无限重试是用 LLM 掩盖数据问题, 成本也不可控。"""
    llm = _StubLLM(["依据 [7]。", "还是依据 [7]。"])
    answer = _generate(llm, _retrieval())

    assert len(llm.calls) == 2
    assert UNVERIFIED_BADGE.format(citations="[7]") in answer


def test_still_bad_answer_is_kept_and_labelled():
    """仍不可核验时**保留答案 + 标注**，而不是丢弃或假装没事。"""
    llm = _StubLLM(["依据 [7]。", "还是依据 [7]。"])
    answer = _generate(llm, _retrieval())

    assert answer.startswith("还是依据")  # 答案还在
    assert "部分依据未能核验" in answer


# ============================================================
# 存在性校验（T-29）接入后同样是"问题"
# ============================================================
def test_missing_authority_row_counts_as_a_problem(monkeypatch):
    """权威表里查不到 -> 也是问题, 走同一条降级路径。"""
    monkeypatch.setattr(db_manager, "existing_law_article_ids", lambda ids: set())
    retrieval = _retrieval(version_id="ver-2018")
    llm = _StubLLM(["根据 [1]。", "根据 [1]。"])

    answer = _generate(llm, retrieval)

    assert len(llm.calls) == 2
    assert UNVERIFIED_BADGE.format(citations="[1]") in answer


def test_present_authority_row_passes(monkeypatch):
    """权威表里有这条 -> 不该触发重生成。"""
    from app.core.generate.citation import authoritative_article_id
    from app.core.law_versions import make_article_id

    monkeypatch.setattr(
        db_manager,
        "existing_law_article_ids",
        lambda ids: {make_article_id("ver-2018", "第四十七条", 46)},
    )
    retrieval = _retrieval(version_id="ver-2018")
    llm = _StubLLM(["根据《劳动合同法》第四十七条 [1]，应按工作年限支付经济补偿。"])

    answer = _generate(llm, retrieval)

    assert len(llm.calls) == 1
    assert "部分依据未能核验" not in answer
    assert authoritative_article_id(retrieval["sources"][0]) == make_article_id(
        "ver-2018", "第四十七条", 46
    )


def test_lookup_failure_fails_open(monkeypatch):
    """查表本身报错时放行 —— 校验组件出问题不该让问答挂掉, 也不该凭空造结论。"""

    def _boom(ids):
        raise RuntimeError("MySQL 挂了")

    monkeypatch.setattr(db_manager, "existing_law_article_ids", _boom)
    retrieval = _retrieval(version_id="ver-2018")
    llm = _StubLLM(["根据 [1] 处理。"])

    answer = _generate(llm, retrieval)

    assert answer == "根据 [1] 处理。"
    assert len(llm.calls) == 1


# ============================================================
# 开关
# ============================================================
def test_switch_off_disables_both_checks(monkeypatch):
    """`ENABLE_CITATION_VERIFY=False` 时两类校验一起停 —— 否则"关掉开关"是伪生效。"""
    monkeypatch.setattr(settings, "ENABLE_CITATION_VERIFY", False)
    llm = _StubLLM(["依据 [7]。"])

    answer = _generate(llm, _retrieval())

    assert answer == "依据 [7]。"
    assert len(llm.calls) == 1


# ============================================================
# 流式: 明确不重生成
# ============================================================
def test_stream_warning_labels_without_regeneration():
    """流式只能标注、不能重生成(错误引用已经发给用户了), 这条锁住这个决定。"""
    warning = chat_service._citation_warning("依据 [7]。", _retrieval())
    assert "部分依据未能核验" in warning
    assert "[7]" in warning


def test_stream_warning_is_empty_when_clean():
    assert chat_service._citation_warning("根据 [1] 处理。", _retrieval()) == ""
