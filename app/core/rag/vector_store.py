"""
向量存储模块 - 使用 Milvus 存储和检索向量

新增能力(都是 P0 任务的支撑):
    iter_chunks()         从 Milvus 读回全部正文与元数据
                          —— 解决重启后 BM25 语料为空的问题(T-10),
                             也解决了"检索到的片段必须能回查原文"的追溯需求。
    rebuild_bm25_nodes()  用上面的读回结果重建 BM25 语料
    delete_by_document()  按文档删除向量, 避免留下永远删不掉的孤儿向量(F1.9)

关于元数据过滤的一点重要说明(已知缺口, 记录在案):
    LlamaIndex 的 MilvusVectorStore 把节点元数据序列化进 `_node_content` 字段,
    并不会为 `tenant_id` / `validity_status` 这类键自动建标量字段。
    这意味着"把时效过滤下推到 Milvus 查询表达式"这个设计目标**当前做不到**,
    只能先按 `_node_content` 在应用层过滤(本文件 iter_chunks 的做法)。
    要真正下推, 需要在建集合时显式声明这些标量字段 —— 列为 P2 任务,
    且必须在 Milvus 真跑起来的环境中验证, 不能靠猜。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore

from app.config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Milvus 向量存储管理器"""

    def __init__(self, collection_name: str = None, embed_model=None):
        self.collection_name = collection_name or settings.MILVUS_COLLECTION_NAME
        self._embed_model = embed_model
        self._vector_store: Optional[MilvusVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None

    @property
    def vector_store(self) -> MilvusVectorStore:
        """懒加载向量存储"""
        if self._vector_store is None:
            logger.info("[VectorStore] 连接 Milvus: %s", settings.milvus_uri)
            self._vector_store = MilvusVectorStore(
                uri=settings.milvus_uri,
                collection_name=self.collection_name,
                dim=settings.EMBEDDING_DIM,
                overwrite=False,
            )
        return self._vector_store

    def create_index(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """创建/追加向量索引。"""
        logger.info("[VectorStore] 写入索引, 共 %d 个节点", len(nodes))
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self._embed_model,
            show_progress=False,
        )
        return self._index

    def load_index(self) -> VectorStoreIndex:
        """加载已有索引(服务启动时)。"""
        logger.info("[VectorStore] 从 Milvus 加载索引: %s", self.collection_name)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self._embed_model,
        )
        return self._index

    def get_index(self) -> Optional[VectorStoreIndex]:
        return self._index

    # ============================================================
    # 读回正文 (T-10 / 追溯需求)
    # ============================================================
    def iter_chunks(self, batch_size: int = 500) -> List[Dict[str, Any]]:
        """读回集合中的全部片段。

        Returns:
            [{"id": str, "text": str, "metadata": dict}, ...]

        实现说明: 直接走 pymilvus, 因为 LlamaIndex 没有暴露"列出全部节点"的接口。
        正文与元数据都在 `_node_content` 里(JSON 序列化的节点快照), 需要自己解。
        """
        from pymilvus import Collection, connections, utility

        connections.connect(uri=settings.milvus_uri, alias="default")
        try:
            if not utility.has_collection(self.collection_name):
                logger.warning("[VectorStore] 集合不存在: %s", self.collection_name)
                return []

            collection = Collection(self.collection_name)
            output_fields = ["id", "_node_content", "_node_type", "doc_id"]
            available = {field.name for field in collection.schema.fields}
            output_fields = [name for name in output_fields if name in available]
            if "_node_content" not in available:
                logger.warning(
                    "[VectorStore] 集合中找不到 _node_content 字段(现有字段: %s), "
                    "无法读回正文 —— BM25 将不可用",
                    sorted(available),
                )
                return []

            records: List[Dict[str, Any]] = []
            # 用 query_iterator 分页, 避免一次性把大集合拉进内存
            iterator = collection.query_iterator(
                batch_size=batch_size,
                expr='id != ""',
                output_fields=output_fields,
            )
            while True:
                batch = iterator.next()
                if not batch:
                    iterator.close()
                    break
                for row in batch:
                    parsed = _parse_node_row(row)
                    if parsed:
                        records.append(parsed)
            logger.info("[VectorStore] 读回片段 %d 条", len(records))
            return records
        finally:
            connections.disconnect("default")

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """按片段 ID 取回原文与元数据。

        这是"引用可核验"的技术兑现点: 回答里的每条引用都能用这个接口
        取到当时喂给模型的那段原文, 而不是让用户去相信一个转述。
        """
        from pymilvus import Collection, connections, utility

        connections.connect(uri=settings.milvus_uri, alias="default")
        try:
            if not utility.has_collection(self.collection_name):
                return None
            collection = Collection(self.collection_name)
            rows = collection.query(
                expr=f'id == "{chunk_id}"',
                output_fields=["id", "_node_content"],
            )
            return _parse_node_row(rows[0]) if rows else None
        finally:
            connections.disconnect("default")

    def rebuild_bm25_nodes(self) -> List[TextNode]:
        """用 Milvus 中读回的正文重建 BM25 语料节点。"""
        nodes: List[TextNode] = []
        for record in self.iter_chunks():
            nodes.append(
                TextNode(
                    id_=record["id"],
                    text=record["text"],
                    metadata=record.get("metadata") or {},
                )
            )
        return nodes

    # ============================================================
    # 删除 (F1.9)
    # ============================================================
    def delete_by_document(self, document_id: str) -> int:
        """删除某个文档的全部向量, 返回删除条数。

        因为元数据没有独立的标量字段(见模块文档说明), 只能先读回再按
        主键删除。数据量大时这会是性能瓶颈, 属于 P2 要解决的已知缺口。
        """
        from pymilvus import Collection, connections, utility

        targets = [
            record["id"]
            for record in self.iter_chunks()
            if str((record.get("metadata") or {}).get("document_id", "")) == str(document_id)
        ]
        if not targets:
            logger.info("[VectorStore] 文档 %s 没有需要清理的向量", document_id)
            return 0

        connections.connect(uri=settings.milvus_uri, alias="default")
        try:
            if not utility.has_collection(self.collection_name):
                return 0
            collection = Collection(self.collection_name)
            collection.delete(expr=f'id in {json.dumps(targets)}')
            collection.flush()
            logger.info("[VectorStore] 文档 %s 已删除 %d 条向量", document_id, len(targets))
            return len(targets)
        finally:
            connections.disconnect("default")

    def drop_collection(self) -> None:
        """删除集合(仅测试用)。"""
        from pymilvus import Collection, connections, utility

        connections.connect(uri=settings.milvus_uri, alias="default")
        try:
            if utility.has_collection(self.collection_name):
                Collection(self.collection_name).drop()
                logger.info("[VectorStore] 已删除集合: %s", self.collection_name)
        finally:
            connections.disconnect("default")


def _parse_node_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把 Milvus 的一行还原成 {id, text, metadata}。"""
    raw = row.get("_node_content")
    if not raw:
        return None
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError) as exc:
        logger.warning("[VectorStore] 无法解析 _node_content (%s): %s", row.get("id"), exc)
        return None

    text = payload.get("text") or ""
    metadata = payload.get("metadata") or {}
    if not text:
        return None
    return {
        "id": row.get("id") or payload.get("id_"),
        "text": text,
        "metadata": metadata,
    }
