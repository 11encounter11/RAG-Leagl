"""
向量化模块 - 将文本转换为向量
使用智谱 Embedding API（无需本地下载模型）
"""
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

from app.config.settings import settings


class Embedder:
    """文本向量化器 - 使用智谱 embedding-3 模型"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.ZHIPU_API_KEY
        self._embed_model = None

    @property
    def embed_model(self) -> ZhipuAIEmbedding:
        """懒加载 embedding 模型"""
        if self._embed_model is None:
            print("[Embedder] 初始化智谱 Embedding API (embedding-3)")
            self._embed_model = ZhipuAIEmbedding(
                api_key=self.api_key,
                model="embedding-3",
            )
        return self._embed_model

    def get_text_embedding(self, text: str) -> list[float]:
        """获取单条文本的向量"""
        return self.embed_model.get_text_embedding(text)

    def get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本向量"""
        return self.embed_model.get_text_embedding_batch(texts)
