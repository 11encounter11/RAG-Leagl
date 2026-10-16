"""
LLM 基类 - 定义统一的模型调用接口
所有模型实现都必须继承此类，保证接口一致
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple, Union


@dataclass
class LLMUsage:
    """一次生成消耗的量（需求 F8.3）。

    刻意**不放在 LLM 实例的属性上**: `llm_router` 只持有一个共享实例(见 router.py),
    并发问答会互相覆盖, 于是"A 的答案记了 B 的 token"。用量必须跟着这次调用返回。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# `chat_stream_with_usage` 的元素: 文本片段, 或流末尾附上的一次用量。
StreamItem = Union[str, LLMUsage]


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

    async def chat_with_usage(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[str, Optional[LLMUsage]]:
        """非流式对话, 同时带回本次用量（需求 F8.3 / 落库字段 tokens_used）。

        默认实现退化成 `chat()` + 用量 None —— 没有实现它的适配器不会坏, 也只是
        拿不到数。**退化必须是显式的 None 而不是 0**: 记 0 会让人以为"这次没花钱",
        而真实情况是"这个模型没回报用量", 两件事差别很大。
        """
        return await self.chat(messages, **kwargs), None

    async def chat_stream_with_usage(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[StreamItem, None]:
        """流式对话, 末尾额外 yield 一个 `LLMUsage`。

        默认实现只转发文本片段、不产出用量, 同上。
        """
        async for token in self.chat_stream(messages, **kwargs):
            yield token

    def build_system_prompt(self, context: str = "") -> Dict[str, str]:
        """
        构建系统提示词

        委派给 app/core/generate/prompt.py —— 那里是提示词与引用格式的唯一定义处。

        修复 T-01: 原来这里硬编码"必须引用参考资料，使用 [来源: 文件名] 格式标注"，
        而检索上下文里给的编号是 "[1] 来源: xxx"，两边格式对不上。模型面对矛盾
        的要求只能自己发明格式，于是引用既不稳定也无法被前端解析回片段。
        """
        from app.core.generate.prompt import build_system_prompt

        return build_system_prompt(context)
