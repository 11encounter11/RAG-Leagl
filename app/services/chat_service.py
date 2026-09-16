"""
聊天服务 - 核心聊天逻辑，整合 RAG 检索 + LLM 生成
"""
import time
import uuid
from typing import AsyncGenerator, Dict, List

from app.core.llm.router import llm_router
from app.core.memory.redis_store import redis_store
from app.core.memory.db_store import db_manager
from app.services.rag_service import rag_service
from app.utils.sse import sse_format, sse_done, sse_error


class ChatService:
    """聊天服务 - 整合 RAG + LLM 的完整聊天流程"""

    async def chat(
        self,
        message: str,
        model_name: str,
        conversation_id: str = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> dict:
        """
        非流式聊天
        Returns:
            包含回答、会话ID等信息的字典
        """
        start_time = time.time()

        # 1. 获取或创建会话
        if not conversation_id or conversation_id == "null":
            conv = db_manager.create_conversation(title=message[:30], model_name=model_name)
            conversation_id = conv.id
        else:
            conv = db_manager.get_conversation(conversation_id)
            if not conv:
                raise ValueError(f"会话不存在: {conversation_id}")

        # 2. RAG 检索
        nodes, context = rag_service.retrieve(message, top_k, similarity_threshold)

        # 3. 构建消息列表
        llm = llm_router.get(model_name)
        messages = self._build_messages(message, context, conversation_id)

        # 4. 调用 LLM
        response_text = await llm.chat(messages)

        # 5. 保存消息
        db_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=message,
        )
        msg = db_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
            model_name=model_name,
        )

        # 6. 更新 Redis 缓存
        await redis_store.append_message(conversation_id, {"role": "user", "content": message})
        await redis_store.append_message(conversation_id, {"role": "assistant", "content": response_text})

        response_time = time.time() - start_time

        return {
            "conversation_id": conversation_id,
            "message_id": msg.id,
            "content": response_text,
            "model_name": model_name,
            "retrieval_count": len(nodes),
            "response_time": round(response_time, 2),
        }

    async def chat_stream(
        self,
        message: str,
        model_name: str,
        conversation_id: str = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> AsyncGenerator[str, None]:
        """
        流式聊天（SSE）
        Yields:
            SSE 格式的数据
        """
        # 1. 获取或创建会话
        if not conversation_id or conversation_id == "null":
            conv = db_manager.create_conversation(title=message[:30], model_name=model_name)
            conversation_id = conv.id
        else:
            conv = db_manager.get_conversation(conversation_id)
            if not conv:
                yield await sse_error(f"会话不存在: {conversation_id}")
                return

        # 2. RAG 检索
        nodes, context = rag_service.retrieve(message, top_k, similarity_threshold)

        # 3. 构建消息列表
        llm = llm_router.get(model_name)
        messages = self._build_messages(message, context, conversation_id)

        # 4. 流式调用 LLM
        full_response = ""
        try:
            async for token in llm.chat_stream(messages):
                full_response += token
                yield await sse_format({
                    "type": "token",
                    "content": token,
                    "conversation_id": conversation_id,
                    "model_name": model_name,
                    "retrieval_count": len(nodes),
                })
        except Exception as e:
            yield await sse_error(str(e))
            return

        # 5. 保存消息
        db_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=message,
        )
        db_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_response,
            model_name=model_name,
        )

        # 6. 更新 Redis 缓存
        await redis_store.append_message(conversation_id, {"role": "user", "content": message})
        await redis_store.append_message(conversation_id, {"role": "assistant", "content": full_response})

        # 7. 发送结束信号
        yield await sse_done(conversation_id, model_name)

    def _build_messages(
        self,
        user_message: str,
        context: str,
        conversation_id: str,
    ) -> List[Dict[str, str]]:
        """
        构建发送给 LLM 的消息列表
        包含：系统提示词 + 历史对话 + 当前问题
        """
        llm = llm_router.get("glm-5.3-flash")  # 用任意实例获取 system prompt
        messages = [llm.build_system_prompt(context)]

        # 从数据库获取历史对话
        db_messages = db_manager.get_messages(conversation_id, limit=20)
        history = [{"role": m.role, "content": m.content} for m in db_messages]

        # 添加历史对话（最近 10 条）
        for msg in history[-10:]:
            messages.append(msg)

        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})

        return messages


# 全局实例
chat_service = ChatService()
