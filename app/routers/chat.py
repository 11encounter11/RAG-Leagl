"""
聊天路由 - 聊天接口
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import chat_service

router = APIRouter(prefix="/api/chat", tags=["聊天"])


@router.post("", response_model=ChatResponse, summary="非流式聊天")
async def chat(request: ChatRequest):
    """
    发送消息并获取完整回答（非流式）
    - 自动进行 RAG 检索
    - 支持多模型切换
    - 支持多轮对话
    """
    try:
        result = await chat_service.chat(
            message=request.message,
            model_name=request.model_name.value,
            conversation_id=request.conversation_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            use_agentic_rag=request.use_agentic_rag,
        )
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天失败: {str(e)}")


@router.post("/stream", summary="流式聊天（SSE）")
async def chat_stream(request: ChatRequest):
    """
    发送消息并以 SSE 流式获取回答
    - Token 级流式输出
    - 支持前端实时渲染
    """
    try:
        return StreamingResponse(
            chat_service.chat_stream(
                message=request.message,
                model_name=request.model_name.value,
                conversation_id=request.conversation_id,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                use_agentic_rag=request.use_agentic_rag,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流式聊天失败: {str(e)}")
