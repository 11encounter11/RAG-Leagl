"""
向量化模块 - 将文本转换为向量
使用智谱 Embedding API（无需本地下载模型）
"""
import logging

from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

from app.config.settings import settings

logger = logging.getLogger(__name__)

# 实际请求的模型名, **这里是唯一事实来源**。
# 注意 `settings.EMBEDDING_MODEL_NAME` 并没有被本模块读取 —— `.env` 里那行
# `BAAI/bge-large-zh-v1.5` 是切到智谱之前留下的死配置, 改了它也不会生效。
# 留痕(T-31)因此不能去读 settings, 否则会把没用过的模型记成用过的。
EMBEDDING_MODEL = "embedding-3"


class Embedder:
    """文本向量化器 - 使用智谱 embedding-3 模型"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.ZHIPU_API_KEY
        self._embed_model = None

    @property
    def embed_model(self) -> ZhipuAIEmbedding:
        """懒加载 embedding 模型"""
        if self._embed_model is None:
            logger.info("[Embedder] 初始化智谱 Embedding API (%s)", EMBEDDING_MODEL)
            self._embed_model = ZhipuAIEmbedding(
                api_key=self.api_key,
                model=EMBEDDING_MODEL,
            )
        return self._embed_model

    def get_text_embedding(self, text: str) -> list[float]:
        """获取单条文本的向量"""
        return self.embed_model.get_text_embedding(text)

    def get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本向量"""
        return self.embed_model.get_text_embedding_batch(texts)
