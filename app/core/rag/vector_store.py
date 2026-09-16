"""
向量存储模块 - 使用 Milvus 存储和检索向量
"""
from typing import List, Optional
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

from app.config.settings import settings


class VectorStoreManager:
    """Milvus 向量存储管理器"""

    def __init__(self, collection_name: str = None, embed_model=None):
        self.collection_name = collection_name or settings.MILVUS_COLLECTION_NAME
        self._embed_model = embed_model
        self._vector_store = None
        self._index = None

    @property
    def vector_store(self) -> MilvusVectorStore:
        """懒加载向量存储"""
        if self._vector_store is None:
            print(f"[VectorStore] 连接 Milvus: {settings.milvus_uri}")
            self._vector_store = MilvusVectorStore(
                uri=settings.milvus_uri,
                collection_name=self.collection_name,
                dim=2048,  # 智谱 embedding-3 的向量维度
                overwrite=False,
            )
        return self._vector_store

    def create_index(self, nodes: list[TextNode]) -> VectorStoreIndex:
        """
        创建向量索引（用于首次导入文档）
        Args:
            nodes: 分块后的文本节点列表
        Returns:
            VectorStoreIndex 实例
        """
        print(f"[VectorStore] 创建索引，共 {len(nodes)} 个节点")
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self._embed_model,
            show_progress=True,
        )
        print(f"[VectorStore] 索引创建完成")
        return self._index

    def load_index(self) -> VectorStoreIndex:
        """
        加载已有索引（用于服务启动时）
        Returns:
            VectorStoreIndex 实例
        """
        print(f"[VectorStore] 从 Milvus 加载索引: {self.collection_name}")
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self._embed_model,
        )
        print(f"[VectorStore] 索引加载完成")
        return self._index

    def get_index(self) -> Optional[VectorStoreIndex]:
        """获取当前索引"""
        return self._index

    def drop_collection(self):
        """删除集合（用于测试）"""
        from pymilvus import connections, Collection
        connections.connect(uri=settings.milvus_uri)
        if Collection(self.collection_name).exists():
            Collection(self.collection_name).drop()
            print(f"[VectorStore] 已删除集合: {self.collection_name}")
        connections.disconnect("default")
