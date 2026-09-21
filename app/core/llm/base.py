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

        委派给 app/core/generate/prompt.py —— 那里是提示词与引用格式的唯一定义处。

        修复 T-01: 原来这里硬编码"必须引用参考资料，使用 [来源: 文件名] 格式标注"，
        而检索上下文里给的编号是 "[1] 来源: xxx"，两边格式对不上。模型面对矛盾
        的要求只能自己发明格式，于是引用既不稳定也无法被前端解析回片段。
        """
        from app.core.generate.prompt import build_system_prompt

        return build_system_prompt(context)
