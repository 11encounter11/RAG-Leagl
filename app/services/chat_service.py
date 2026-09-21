"""
聊天服务 - 核心聊天逻辑, 整合 Agentic RAG 工作流 + LLM 生成

本次改动:
  1. 抽出 _retrieve() —— 原来 chat() 和 chat_stream() 各写了一遍完全相同的
     检索分支, 改一处必须记得改另一处, 是典型的分叉隐患。
  2. 系统提示词改从 app/core/generate/prompt.py 取(T-01), 不再经 LLM 实例兜一圈。
  3. 支持拒答: 工作流判定依据不足时直接返回固定文案, **不调用 LLM**。
     这样"不编造"由架构保证, 而不是靠提示词祈祷。
"""
from __future__ import annotations

import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

from app.core.generate.prompt import build_system_prompt
from app.core.llm.router import llm_router
from app.core.memory.db_store import db_manager
from app.core.memory.redis_store import redis_store
from app.services.rag_service import rag_service
from app.utils.sse import sse_done, sse_error, sse_format

logger = logging.getLogger(__name__)

# 拒答文案。写成常量而不是交给模型自由发挥: 依据不足时最危险的动作就是
# 让模型"尽力回答", 那正是幻觉的来源。
REFUSAL_MESSAGE = (
    "抱歉，我在现有知识库中没有找到足以回答这个问题的法律依据，因此不能给出结论。\n\n"
    "为了避免给你错误的信息，请补充一些关键信息，例如：\n"
    "- 事情发生在哪个省市（法律适用有地域差异）；\n"
    "- 涉及的是哪一类纠纷（劳动合同、买卖、租赁、婚姻家庭等）；\n"
    "- 关键事实与时间点（例如在职时长、是否签了书面合同、被通知的时间）。\n\n"
    "补充后我再帮你查一次。"
)


def _chunk_text(text: str, size: int = 24) -> List[str]:
    """把固定文案切成小片, 让拒答在前端也呈现为打字机效果。"""
    return [text[i : i + size] for i in range(0, len(text), size)]


class ChatService:
    """聊天服务 - 整合 Agentic RAG + LLM 的完整聊天流程"""

    # ============================================================
    # 检索
    # ============================================================
    async def _retrieve(
        self,
        message: str,
        top_k: int,
        similarity_threshold: float,
        use_agentic_rag: bool,
    ) -> Dict:
        """统一的检索入口, 供流式与非流式共用。"""
        if use_agentic_rag:
            result = await rag_service.retrieve_with_workflow(
                message, top_k, similarity_threshold
            )
            return {
                "nodes": result.get("reranked_docs", []),
                "context": result.get("context", ""),
                "sources": result.get("sources", []),
                "workflow_steps": result.get("workflow_steps", []),
                "reflection_score": result.get("reflection_score"),
                "reflection_note": result.get("reflection_note", ""),
                "refused": bool(result.get("refuse", False)),
                "refuse_reason": result.get("refuse_reason", ""),
                "domain_score": result.get("domain_score"),
                "confidence_level": result.get("confidence_level", "high"),
            }

        nodes, context = rag_service.retrieve(message, top_k, similarity_threshold)
        return {
            "nodes": nodes,
            "context": context,
            "sources": rag_service.get_sources_info(nodes),
            "workflow_steps": ["retrieval:direct"],
            "reflection_score": None,
            "reflection_note": "",
            "refused": not nodes,
            "refuse_reason": "" if nodes else "直接检索模式未召回任何片段",
            "domain_score": None,
            "confidence_level": "high" if nodes else "out_of_domain",
        }

    @staticmethod
    def _with_caution(answer: str, retrieval: Dict) -> str:
        """低置信时在回答末尾追加提示。

        刻意写成"追加"而不是"拦截": 匹配度偏低不等于没有依据, 直接拒答会让
        正常问题也得不到帮助。但用户必须知道这条依据的可信度不高 ——
        把不确定性藏起来, 才是法律场景里最危险的做法。
        """
        if retrieval.get("confidence_level") != "low":
            return answer
        score = retrieval.get("domain_score")
        shown = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        note = (
            f"⚠️ 提示：本次检索与知识库的匹配度偏低（领域判据 {shown}），"
            f"以上依据可能不完全适用于你描述的场景，请务必点开原文核对。"
        )
        # 流式场景会以空 answer 调用本方法(只取提示文案), 此时不要多出空行
        return f"{answer}\n\n---\n{note}" if answer else f"\n\n---\n{note}"

    def _build_messages(
        self,
        user_message: str,
        context: str,
        conversation_id: str,
    ) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息列表: 系统提示词 + 历史对话 + 当前问题。"""
        messages: List[Dict[str, str]] = [build_system_prompt(context)]

        db_messages = db_manager.get_messages(conversation_id, limit=20)
        for msg in db_messages[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_message})
        return messages

    def _persist(
        self, conversation_id: str, message: str, response_text: str, model_name: str
    ) -> Optional[int]:
        """落库。"""
        db_manager.add_message(conversation_id=conversation_id, role="user", content=message)
        saved = db_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
            model_name=model_name,
        )
        return getattr(saved, "id", None)

    def _resolve_conversation(
        self, message: str, model_name: str, conversation_id: Optional[str]
    ):
        if not conversation_id or conversation_id == "null":
            return db_manager.create_conversation(title=message[:30], model_name=model_name)
        conv = db_manager.get_conversation(conversation_id)
        if not conv:
            raise ValueError(f"会话不存在: {conversation_id}")
        return conv

    # ============================================================
    # 非流式
    # ============================================================
    async def chat(
        self,
        message: str,
        model_name: str,
        conversation_id: str = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        use_agentic_rag: bool = True,
    ) -> dict:
        """非流式聊天。"""
        start_time = time.time()

        conv = self._resolve_conversation(message, model_name, conversation_id)
        conversation_id = conv.id

        retrieval = await self._retrieve(message, top_k, similarity_threshold, use_agentic_rag)
        nodes = retrieval["nodes"]

        # 依据不足 -> 确定性拒答, 不调用 LLM(见 REFUSAL_MESSAGE 的注释)
        if retrieval["refused"]:
            response_text = REFUSAL_MESSAGE
            logger.info(
                "[ChatService] 依据不足, 返回拒答文案(未调用 LLM): %s",
                retrieval.get("refuse_reason", ""),
            )
        else:
            llm = llm_router.get(model_name)
            messages = self._build_messages(message, retrieval["context"], conversation_id)
            response_text = self._with_caution(await llm.chat(messages), retrieval)

        message_id = self._persist(conversation_id, message, response_text, model_name)

        await redis_store.append_message(conversation_id, {"role": "user", "content": message})
        await redis_store.append_message(
            conversation_id, {"role": "assistant", "content": response_text}
        )

        return {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "content": response_text,
            "model_name": model_name,
            "retrieval_count": len(nodes),
            "response_time": round(time.time() - start_time, 2),
            "sources": retrieval["sources"],
            "workflow_steps": retrieval["workflow_steps"],
            "reflection_score": retrieval["reflection_score"],
            "reflection_note": retrieval["reflection_note"],
            "refused": retrieval["refused"],
            "refuse_reason": retrieval.get("refuse_reason", ""),
            "domain_score": retrieval.get("domain_score"),
            "confidence_level": retrieval.get("confidence_level", "high"),
        }

    # ============================================================
    # 流式
    # ============================================================
    async def chat_stream(
        self,
        message: str,
        model_name: str,
        conversation_id: str = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        use_agentic_rag: bool = True,
    ) -> AsyncGenerator[str, None]:
        """流式聊天(SSE)。"""
        try:
            conv = self._resolve_conversation(message, model_name, conversation_id)
        except ValueError as exc:
            yield await sse_error(str(exc))
            return
        conversation_id = conv.id

        retrieval = await self._retrieve(message, top_k, similarity_threshold, use_agentic_rag)
        nodes = retrieval["nodes"]
        full_response = ""

        if retrieval["refused"]:
            logger.info("[ChatService] 依据不足, 流式返回拒答文案(未调用 LLM)")
            for chunk in _chunk_text(REFUSAL_MESSAGE):
                full_response += chunk
                yield await sse_format(
                    {
                        "type": "token",
                        "content": chunk,
                        "conversation_id": conversation_id,
                        "model_name": model_name,
                        "retrieval_count": 0,
                    }
                )
        else:
            llm = llm_router.get(model_name)
            messages = self._build_messages(message, retrieval["context"], conversation_id)
            try:
                async for token in llm.chat_stream(messages):
                    full_response += token
                    yield await sse_format(
                        {
                            "type": "token",
                            "content": token,
                            "conversation_id": conversation_id,
                            "model_name": model_name,
                            "retrieval_count": len(nodes),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("[ChatService] 流式生成失败")
                yield await sse_error(str(exc))
                return

            # 低置信提示在流结束后追加(见 _with_caution 的说明)
            caution = self._with_caution("", retrieval)
            if caution:
                full_response += caution
                yield await sse_format(
                    {
                        "type": "token",
                        "content": caution,
                        "conversation_id": conversation_id,
                        "model_name": model_name,
                        "retrieval_count": len(nodes),
                    }
                )

        # 来源信息总是发一次, 前端不必区分"没有来源"和"字段没来"
        yield await sse_format(
            {
                "type": "sources",
                "sources": retrieval["sources"],
                "conversation_id": conversation_id,
                "workflow_steps": retrieval["workflow_steps"],
                "reflection_score": retrieval["reflection_score"],
                "reflection_note": retrieval["reflection_note"],
                "refused": retrieval["refused"],
                "refuse_reason": retrieval.get("refuse_reason", ""),
                "domain_score": retrieval.get("domain_score"),
                "confidence_level": retrieval.get("confidence_level", "high"),
            }
        )

        self._persist(conversation_id, message, full_response, model_name)
        await redis_store.append_message(conversation_id, {"role": "user", "content": message})
        await redis_store.append_message(
            conversation_id, {"role": "assistant", "content": full_response}
        )

        yield await sse_done(conversation_id, model_name)


# 全局实例
chat_service = ChatService()
