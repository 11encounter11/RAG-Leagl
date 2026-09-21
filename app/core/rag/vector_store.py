"""
向量存储模块 - 使用 Milvus 存储和检索向量

本模块承担三个测试期才暴露出来的关键约束(见 docs/04-P0任务清单.md N-5/N-7/N-8):

  1. **维度必须实测, 不能写死**
     原代码把 dim 硬编码成 2048, 而智谱 embedding-3 实际返回 1024 维。
     这个不一致不会在导入时报错, 只在真正 insert 时由 Milvus 抛出
     `the dim of field data is not equal to schema dim` —— 到那时
     107 次嵌入调用已经白花掉了。所以: 默认自动探测 + 建库前校验。

  2. **元数据必须显式声明为标量字段**
     被测版本(llama-index-vector-stores-milvus==1.2.0)的集合布局是
     `id / doc_id / text / embedding`, 元数据不会自动落库。
     不声明 scalar_field_names 的后果是 article_no / validity_status /
     tenant_id 入库即丢失, 而检索依然能返回文本 —— 故障完全静默。
     声明之后还有一个额外收益: 这些字段成为真实标量字段,
     "把时效/租户过滤下推到查询表达式"才有了实现基础。

  3. **Milvus 不接受缺失字段**
     已声明的标量字段在每条记录里都必须出现且类型匹配, 缺一个整批失败。
     所以入库前用 ensure_scalar_defaults 统一补齐。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import DataType

from app.config.settings import settings
from app.core.ingest.metadata import (
    MILVUS_RESERVED_FIELDS,
    ensure_scalar_defaults,
    scalar_fields_for_milvus,
)

logger = logging.getLogger(__name__)

# 契约层用字符串描述类型, 这里映射成 Milvus 的 DataType
_TYPE_MAP = {
    "VARCHAR": DataType.VARCHAR,
    "INT64": DataType.INT64,
    "BOOL": DataType.BOOL,
    "DOUBLE": DataType.DOUBLE,
    "FLOAT": DataType.FLOAT,
}


def _milvus_types() -> List[DataType]:
    _, type_names = scalar_fields_for_milvus()
    missing = sorted({name for name in type_names if name not in _TYPE_MAP})
    if missing:
        raise ValueError(f"未知的标量字段类型: {missing}")
    return [_TYPE_MAP[name] for name in type_names]


class VectorStoreManager:
    """Milvus 向量存储管理器"""

    def __init__(self, collection_name: str = None, embed_model=None):
        self.collection_name = collection_name or settings.MILVUS_COLLECTION_NAME
        self._embed_model = embed_model
        self._vector_store: Optional[MilvusVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None
        self._dim: Optional[int] = None

    # ============================================================
    # 维度
    # ============================================================
    @property
    def dim(self) -> int:
        """向量维度。配置为 0 时向嵌入模型实测一次并缓存。"""
        if self._dim is None:
            if settings.EMBEDDING_DIM > 0:
                self._dim = int(settings.EMBEDDING_DIM)
            else:
                if self._embed_model is None:
                    raise RuntimeError("无法探测向量维度: 没有可用的嵌入模型")
                probe = self._embed_model.get_text_embedding("维度探测")
                self._dim = len(probe)
                logger.info("[VectorStore] 实测向量维度 = %d", self._dim)
        return self._dim

    def _existing_collection_dim(self) -> Optional[int]:
        """读已存在集合的向量维度; 集合不存在时返回 None。"""
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus_uri)
        if self.collection_name not in client.list_collections():
            return None
        description = client.describe_collection(collection_name=self.collection_name)
        for field in description.get("fields", []):
            if field.get("params", {}).get("dim"):
                return int(field["params"]["dim"])
        return None

    def verify_collection_compatible(self) -> None:
        """建库前校验: 已存在集合的维度必须与实际嵌入维度一致。

        刻意在**嵌入之前**抛错。否则会先老老实实调 107 次嵌入 API,
        再让 Milvus 在 insert 阶段报错 —— 时间和钱都白花了。
        """
        existing = self._existing_collection_dim()
        if existing is None:
            return
        if existing != self.dim:
            raise RuntimeError(
                f"集合 {self.collection_name} 的向量维度是 {existing}, "
                f"而当前嵌入模型输出 {self.dim} 维, 二者不一致。\n"
                f"处理办法(择一):\n"
                f"  1. 换用维度匹配的嵌入模型;\n"
                f"  2. 若确认当前模型正确, 把旧集合改名留档后重建:\n"
                f"     python -c \"from app.core.rag.vector_store import VectorStoreManager as V;"
                f"V('{self.collection_name}').rename_collection('{self.collection_name}_old')\"\n"
                f"  3. 在 .env 中显式设 EMBEDDING_DIM={existing} 以锁定旧维度(仅当旧集合仍要沿用)"
            )

    # ============================================================
    # 存储
    # ============================================================
    @property
    def vector_store(self) -> MilvusVectorStore:
        """懒加载向量存储。"""
        if self._vector_store is None:
            self.verify_collection_compatible()
            names, _ = scalar_fields_for_milvus()
            collisions = [name for name in names if name in MILVUS_RESERVED_FIELDS]
            if collisions:
                raise ValueError(
                    f"标量字段名与 Milvus 保留字段冲突: {collisions}; "
                    f"保留字段为 {MILVUS_RESERVED_FIELDS}"
                )
            logger.info(
                "[VectorStore] 连接 Milvus %s, 集合=%s, dim=%d, 标量字段=%d 个",
                settings.milvus_uri,
                self.collection_name,
                self.dim,
                len(names),
            )
            self._vector_store = MilvusVectorStore(
                uri=settings.milvus_uri,
                collection_name=self.collection_name,
                dim=self.dim,
                overwrite=False,
                # 元数据落库的关键(见模块文档第 2 条)
                scalar_field_names=names,
                scalar_field_types=_milvus_types(),
            )
        return self._vector_store

    def create_index(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """创建/追加向量索引。"""
        for node in nodes:
            # 补齐标量字段: Milvus 要求每条记录都带上全部已声明字段
            node.metadata = ensure_scalar_defaults(node.metadata)

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
    # 读回正文与元数据 (T-09 / T-10 / 追溯需求)
    # ============================================================
    def _query_fields(self, collection) -> List[str]:
        """确定可用的输出字段: 固定字段 + 集合里真实存在的标量字段。"""
        available = {field.name for field in collection.schema.fields}
        wanted = ["id", "doc_id", "text"] + list(scalar_fields_for_milvus()[0])
        fields = [name for name in wanted if name in available]
        # 兼容更早版本写入的集合(布局是 _node_content)
        if "_node_content" in available:
            fields.append("_node_content")
        return fields

    def iter_chunks(self, batch_size: int = 500) -> List[Dict[str, Any]]:
        """读回集合中的全部片段。

        Returns:
            [{"id": str, "text": str, "metadata": dict}, ...]
        """
        from pymilvus import Collection, connections, utility

        connections.connect(uri=settings.milvus_uri, alias="default")
        try:
            if not utility.has_collection(self.collection_name):
                logger.warning("[VectorStore] 集合不存在: %s", self.collection_name)
                return []

            collection = Collection(self.collection_name)
            output_fields = self._query_fields(collection)
            if "text" not in output_fields and "_node_content" not in output_fields:
                logger.warning(
                    "[VectorStore] 集合中没有 text 字段(现有字段: %s), 无法读回正文, BM25 将不可用",
                    sorted(field.name for field in collection.schema.fields),
                )
                return []

            records: List[Dict[str, Any]] = []
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
                    parsed = _parse_chunk_row(row)
                    if parsed:
                        records.append(parsed)
            logger.info("[VectorStore] 读回片段 %d 条", len(records))
            return records
        finally:
            connections.disconnect("default")

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """按片段 ID 取回原文与元数据 —— 引用可核验的落点。"""
        from pymilvus import Collection, connections, utility

        connections.connect(uri=settings.milvus_uri, alias="default")
        try:
            if not utility.has_collection(self.collection_name):
                return None
            collection = Collection(self.collection_name)
            rows = collection.query(
                expr=f'id == "{chunk_id}"',
                output_fields=self._query_fields(collection),
            )
            return _parse_chunk_row(rows[0]) if rows else None
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
    # 删除与维护
    # ============================================================
    def delete_by_document(self, document_id: str) -> int:
        """删除某个文档的全部向量, 返回删除条数。

        现在 document_id 是真实的 Milvus 标量字段, 可以直接用表达式删除;
        读回再删只是兜底(应对旧集合)。
        """
        from pymilvus import Collection, connections, utility

        connections.connect(uri=settings.milvus_uri, alias="default")
        try:
            if not utility.has_collection(self.collection_name):
                return 0
            collection = Collection(self.collection_name)

            if "document_id" in {field.name for field in collection.schema.fields}:
                collection.delete(expr=f'document_id == "{document_id}"')
                collection.flush()
                logger.info("[VectorStore] 按 document_id 表达式删除完成: %s", document_id)
                return 1

            targets = [
                record["id"]
                for record in self.iter_chunks()
                if str((record.get("metadata") or {}).get("document_id", "")) == str(document_id)
            ]
            if not targets:
                return 0
            collection.delete(expr=f"id in {json.dumps(targets)}")
            collection.flush()
            logger.info("[VectorStore] 文档 %s 已删除 %d 条向量", document_id, len(targets))
            return len(targets)
        finally:
            connections.disconnect("default")

    def rename_collection(self, new_name: str) -> bool:
        """给集合改名(用于给旧 schema 的集合留档, 而不是直接删掉)。"""
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus_uri)
        if self.collection_name not in client.list_collections():
            return False
        client.rename_collection(self.collection_name, new_name)
        logger.info("[VectorStore] 集合 %s 已改名为 %s", self.collection_name, new_name)
        return True

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


def _parse_chunk_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把 Milvus 的一行还原成 {id, text, metadata}。

    需要兼容两种集合布局:
      - 新布局(本模块写入): text 与各标量字段平铺
      - 旧布局(更早版本):   正文包在 _node_content 的 JSON 里
    """
    metadata: Dict[str, Any] = {}

    raw_content = row.get("_node_content")
    if raw_content:
        try:
            payload = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
            metadata.update(payload.get("metadata") or {})
            text = payload.get("text") or ""
        except (TypeError, ValueError) as exc:
            logger.warning("[VectorStore] _node_content 解析失败(%s): %s", row.get("id"), exc)
            text = ""
    else:
        text = row.get("text") or ""

    if not text:
        return None

    # 平铺的标量字段即为元数据
    for field in scalar_fields_for_milvus()[0]:
        if field in row:
            metadata[field] = row[field]
    if row.get("doc_id"):
        metadata.setdefault("doc_id", row["doc_id"])

    return {"id": row.get("id"), "text": text, "metadata": metadata}
