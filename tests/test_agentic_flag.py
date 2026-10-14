# -*- coding: utf-8 -*-
"""服务端开关 ENABLE_AGENTIC_RAG 的生效验证(需求 F6.3 / 05 TC-F6.3-01)。

存在的原因: 该开关此前只进了工作流的缓存键, 从来没有决定"走不走工作流" ——
把它关掉, 服务端照样跑完整 workflow, 正是需求文档里判为不合格的"伪生效"。
"""
import asyncio
from unittest.mock import patch

from app.config.settings import settings
from app.services import rag_service as rag_service_module
from app.services.chat_service import chat_service


def _run(flag, request_flag):
    calls = {"workflow": 0, "direct": 0}

    async def fake_workflow(*args, **kwargs):
        calls["workflow"] += 1
        return {"reranked_docs": [], "context": "", "sources": []}

    def fake_direct(*args, **kwargs):
        calls["direct"] += 1
        return ([], "")

    with patch.object(settings, "ENABLE_AGENTIC_RAG", flag), patch.object(
        rag_service_module.rag_service, "retrieve_with_workflow", fake_workflow
    ), patch.object(
        rag_service_module.rag_service, "retrieve", fake_direct
    ), patch.object(
        rag_service_module.rag_service, "get_sources_info", lambda nodes: []
    ):
        asyncio.run(chat_service._retrieve("试用期最长多久", 5, 0.1, request_flag))
    return calls


def test_flag_on_and_request_on_uses_workflow():
    assert _run(flag=True, request_flag=True) == {"workflow": 1, "direct": 0}


def test_flag_off_forces_direct_even_if_request_asks_workflow():
    """服务端关掉开关后, 即使请求传 use_agentic_rag=True 也不能走工作流。"""
    assert _run(flag=False, request_flag=True) == {"workflow": 0, "direct": 1}


def test_flag_on_and_request_off_uses_direct():
    assert _run(flag=True, request_flag=False) == {"workflow": 0, "direct": 1}
