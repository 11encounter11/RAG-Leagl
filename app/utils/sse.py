"""
SSE 流式输出工具
"""
import json
from typing import AsyncGenerator


async def sse_format(data: dict) -> str:
    """
    将数据格式化为 SSE 格式
    Args:
        data: 要发送的数据字典
    Returns:
        SSE 格式的字符串
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def sse_error(message: str) -> str:
    """格式化错误消息"""
    return await sse_format({"type": "error", "content": message})


async def sse_done(conversation_id: str = "", model_name: str = "") -> str:
    """格式化结束消息"""
    return await sse_format({
        "type": "done",
        "conversation_id": conversation_id,
        "model_name": model_name,
    })
