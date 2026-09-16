"""
LLM 基类 - 定义统一的模型调用接口
所有模型实现都必须继承此类，保证接口一致
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any


class BaseLLM(ABC):
    """LLM 抽象基类 - 策略模式的核心接口"""

    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.extra_params = kwargs

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        非流式对话
        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}, ...]
            **kwargs: 模型特定参数（temperature, max_tokens 等）
        Returns:
            模型回复的文本内容
        """
        pass

    @abstractmethod
    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式对话（SSE）
        Args:
            messages: 消息列表
            **kwargs: 模型特定参数
        Yields:
            逐个 token 的文本片段
        """
        pass

    def build_system_prompt(self, context: str = "") -> Dict[str, str]:
        """
        构建系统提示词
        Args:
            context: RAG 检索到的上下文
        Returns:
            系统消息字典
        """
        if context:
            return {
                "role": "system",
                "content": (
                    "你是一个专业的法律智能助手。请基于以下参考资料回答用户的法律问题。\n"
                    "如果参考资料中没有相关信息，请明确告知用户。\n"
                    "回答要准确、简洁，并引用相关法律条文。\n\n"
                    f"参考资料：\n{context}"
                ),
            }
        return {
            "role": "system",
            "content": "你是一个专业的法律智能助手，擅长解答各类法律问题。回答要准确、简洁，并引用相关法律条文。",
        }
