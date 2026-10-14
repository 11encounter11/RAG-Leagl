"""
会话路由 - 会话管理接口
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import ConversationInfo, ConversationRenameRequest, MessageInfo
from app.core.memory.db_store import db_manager

router = APIRouter(prefix="/api/conversations", tags=["会话"])


@router.get("", summary="获取会话列表")
async def list_conversations(limit: int = 20, offset: int = 0):
    """获取所有会话列表"""
    convs = db_manager.list_conversations(limit=limit, offset=offset)
    result = []
    for c in convs:
        msg_count = db_manager.get_message_count(c.id)
        result.append(ConversationInfo(
            id=c.id,
            title=c.title,
            model_name=c.model_name,
            message_count=msg_count,
            created_at=c.created_at,
            updated_at=c.updated_at,
        ))
    return result


@router.get("/{conversation_id}/messages", summary="获取会话消息")
async def get_messages(conversation_id: str, limit: int = 50):
    """获取指定会话的消息列表"""
    conv = db_manager.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = db_manager.get_messages(conversation_id, limit=limit)
    return [
        MessageInfo(
            id=m.id,
            role=m.role,
            content=m.content,
            model_name=m.model_name,
            created_at=m.created_at,
        )
        for m in messages
    ]


@router.delete("/{conversation_id}", summary="删除会话")
async def delete_conversation(conversation_id: str):
    """删除指定会话及其所有消息"""
    conv = db_manager.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    db_manager.delete_conversation(conversation_id)
    return {"message": "会话已删除"}


@router.put("/{conversation_id}", summary="重命名会话")
async def rename_conversation(conversation_id: str, payload: ConversationRenameRequest):
    """重命名指定会话(需求 F5.1: 会话的创建/列表/重命名/删除)。"""
    renamed = db_manager.rename_conversation(conversation_id, payload.title)
    if not renamed:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"message": "会话已重命名", "id": conversation_id, "title": payload.title}
