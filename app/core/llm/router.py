"""
LLM 路由器 - 策略模式的核心
根据用户选择的模型名称，返回对应的 LLM 实例
"""
from app.config.settings import settings
from app.core.llm.base import BaseLLM
from app.core.llm.glm import GLMFlashLLM
from app.core.llm.deepseek import DeepSeekFlashLLM


class LLMRouter:
    """
    LLM 策略路由器
    基于策略模式，根据模型名称返回对应的 LLM 实例
    新增模型只需添加新的实现类并注册到 _strategies 字典
    """

    def __init__(self):
        self._strategies: dict[str, BaseLLM] = {
            "glm-5.3-flash": GLMFlashLLM(api_key=settings.ZHIPU_API_KEY),
            "deepseek-v4.1-flash": DeepSeekFlashLLM(api_key=settings.DEEPSEEK_API_KEY),
        }

    def get(self, model_name: str) -> BaseLLM:
        """
        根据模型名称获取对应的 LLM 实例
        Args:
            model_name: 模型名称，如 "glm-5.3-flash" 或 "deepseek-v4.1-flash"
        Returns:
            对应的 LLM 实例
        Raises:
            ValueError: 不支持的模型名称
        """
        if model_name not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise ValueError(
                f"不支持的模型: {model_name}，可选模型: {available}"
            )
        return self._strategies[model_name]

    def list_models(self) -> list[str]:
        """获取所有可用模型名称"""
        return list(self._strategies.keys())


# 全局路由器实例
llm_router = LLMRouter()
