"""
LLM 路由器 - 策略模式的核心
根据用户选择的模型名称，返回对应的 LLM 实例
"""
from app.config.settings import settings
from app.core.llm.base import BaseLLM
from app.core.llm.deepseek import DeepSeekFlashLLM

# 只注册 DeepSeek(2026-10-14 决定)。
# 原因: 智谱账号的额度留给 embedding —— 入库与检索每一步都要转向量, 它才是必需项;
# 生成环节用 DeepSeek 就够, 不必两家都占额度。
# GLMFlashLLM 的实现保留在 app/core/llm/glm.py, 需要时在这里加回一行即可。


class LLMRouter:
    """
    LLM 策略路由器
    基于策略模式，根据模型名称返回对应的 LLM 实例
    新增模型只需添加新的实现类并注册到 _strategies 字典
    """

    def __init__(self):
        self._strategies: dict[str, BaseLLM] = {
            "deepseek-v4.1-flash": DeepSeekFlashLLM(api_key=settings.DEEPSEEK_API_KEY),
        }

    def get(self, model_name: str) -> BaseLLM:
        """
        根据模型名称获取对应的 LLM 实例
        Args:
            model_name: 模型名称，当前仅 "deepseek-v4.1-flash"
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
