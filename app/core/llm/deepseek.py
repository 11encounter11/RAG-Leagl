"""
DeepSeek V4.1 Flash 模型实现
"""
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from openai import AsyncOpenAI

from app.core.llm.base import BaseLLM, LLMUsage, StreamItem


class DeepSeekFlashLLM(BaseLLM):
    """DeepSeek V4.1 Flash 实现"""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(
            model_name="deepseek-flash",
            api_key=api_key,
            **kwargs,
        )
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """非流式对话"""
        answer, _ = await self.chat_with_usage(messages, **kwargs)
        return answer

    async def chat_with_usage(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[str, Optional[LLMUsage]]:
        """非流式对话 + 用量（F8.3）。

        原实现是 `return response.choices[0].message.content` —— 接口其实回了
        `response.usage`，但只抠文字，用量被扔掉，于是 `tokens_used` 恒为 0 且
        在 chat_service 那一层**根本拿不到数**。
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content, _usage_of(response)

    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式对话"""
        async for item in self.chat_stream_with_usage(messages, **kwargs):
            if isinstance(item, str):
                yield item

    async def chat_stream_with_usage(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[LLMUsage, None]:
        """流式对话, 全部片段吐完后额外 yield 一个用量。

        必须显式带 `stream_options={"include_usage": true}` —— OpenAI 兼容接口
        在 `stream=true` 时**默认不回报用量**，不加这个参数就拿不到数。
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage: Optional[LLMUsage] = None
        async for chunk in response:
            # 带 include_usage 时, 最后一个 chunk 的 choices 为空、只带 usage
            if getattr(chunk, "usage", None) is not None:
                usage = _usage_of(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        if usage is not None:
            yield usage


def _usage_of(response) -> Optional[LLMUsage]:
    """把接口返回的 usage 转成 LLMUsage; 没有回报时返回 None(而不是 0)。"""
    raw = getattr(response, "usage", None)
    if raw is None:
        return None
    return LLMUsage(
        prompt_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(raw, "total_tokens", 0) or 0),
    )
