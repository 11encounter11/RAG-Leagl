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
                    "你是一个智能知识库助手。请基于以下参考资料回答用户的问题。\n"
                    "要求：\n"
                    "1. 回答要准确、简洁、有条理\n"
                    "2. 必须引用参考资料，使用 [来源: 文件名] 格式标注\n"
                    "3. 如果参考资料中没有相关信息，请明确告知用户\n"
                    "4. 不要编造参考资料中没有的内容\n\n"
                    f"参考资料：\n{context}"
                ),
            }
        return {
            "role": "system",
            "content": (
                "你是一个智能知识库助手，擅长基于知识库中的文档回答用户的各类问题。"
                "回答要准确、简洁，并在有依据时标注来源。"
            ),
        }
