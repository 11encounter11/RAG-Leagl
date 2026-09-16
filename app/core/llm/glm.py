"""
智谱 GLM-5.3-Flash 模型实现
"""
import json
from typing import AsyncGenerator, Dict, List
from openai import AsyncOpenAI

from app.core.llm.base import BaseLLM


class GLMFlashLLM(BaseLLM):
    """智谱 GLM-5.3-Flash 实现"""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(
            model_name="glm-5.3-flash",
            api_key=api_key,
            **kwargs,
        )
        # 智谱 API 兼容 OpenAI 接口
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """非流式对话"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content

    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式对话"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
