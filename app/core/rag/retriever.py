"""
检索器 - 从向量数据库中检索相关文档片段
"""
from typing import List, Tuple
from llama_index.core.schema import NodeWithScore

from app.config.settings import settings


class Retriever:
    """向量检索器"""

    def __init__(self, index):
        """
        Args:
            index: VectorStoreIndex 实例
        """
        self._index = index

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        similarity_threshold: float = None,
    ) -> List[NodeWithScore]:
        """
        检索与查询最相关的文档片段
        Args:
            query: 用户查询
            top_k: 返回的片段数量
            similarity_threshold: 相似度阈值，低于此值的片段会被过滤
        Returns:
            带分数的节点列表
        """
        top_k = top_k or settings.TOP_K
        similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD

        # 使用 LlamaIndex 的检索器
        retriever = self._index.as_retriever(similarity_top_k=top_k * 2)
        nodes = retriever.retrieve(query)

        # 过滤低于阈值的片段
        filtered_nodes = [
            node for node in nodes if node.score >= similarity_threshold
        ]

        # 取 top_k 个
        result = filtered_nodes[:top_k]

        print(f"[Retriever] 查询: {query[:30]}... → 召回 {len(nodes)} 个, "
              f"过滤后 {len(filtered_nodes)} 个, 返回 {len(result)} 个")

        return result

    def get_context_text(self, nodes: List[NodeWithScore]) -> str:
        """
        将检索到的节点拼接为上下文文本
        Args:
            nodes: 检索结果节点列表
        Returns:
            拼接后的上下文文本
        """
        context_parts = []
        for i, node in enumerate(nodes, 1):
            source = node.node.metadata.get("filename", "未知来源")
            context_parts.append(
                f"[{i}] 来源: {source}\n{node.node.text}"
            )
        return "\n\n---\n\n".join(context_parts)
