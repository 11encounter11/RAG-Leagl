"""
Redis 对话缓存 - 存储最近对话历史，提供快速访问
"""
import json
from typing import List, Dict, Optional
import redis.asyncio as redis

from app.config.settings import settings


class RedisStore:
    """Redis 对话缓存管理器"""

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._ttl = 3600  # 缓存过期时间：1小时

    async def get_client(self) -> redis.Redis:
        """懒加载 Redis 连接"""
        if self._redis is None:
            self._redis = redis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
        return self._redis

    def _key(self, conversation_id: str) -> str:
        """生成 Redis key"""
        return f"rag:conversation:{conversation_id}:messages"

    async def save_messages(
        self, conversation_id: str, messages: List[Dict[str, str]]
    ):
        """
        保存对话消息到 Redis
        Args:
            conversation_id: 会话ID
            messages: 消息列表
        """
        client = await self.get_client()
        key = self._key(conversation_id)
        await client.setex(key, self._ttl, json.dumps(messages, ensure_ascii=False))

    async def get_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        从 Redis 获取对话历史
        Args:
            conversation_id: 会话ID
        Returns:
            消息列表
        """
        client = await self.get_client()
        key = self._key(conversation_id)
        data = await client.get(key)
        if data:
            return json.loads(data)
        return []

    async def append_message(
        self, conversation_id: str, message: Dict[str, str]
    ):
        """
        追加单条消息
        Args:
            conversation_id: 会话ID
            message: 消息字典
        """
        messages = await self.get_messages(conversation_id)
        messages.append(message)
        await self.save_messages(conversation_id, messages)

    async def delete_messages(self, conversation_id: str):
        """删除会话缓存"""
        client = await self.get_client()
        key = self._key(conversation_id)
        await client.delete(key)

    async def ping(self) -> bool:
        """检查 Redis 连接"""
        try:
            client = await self.get_client()
            return await client.ping()
        except Exception:
            return False


# 全局实例
redis_store = RedisStore()
