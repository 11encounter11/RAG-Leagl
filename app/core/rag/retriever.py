"""
混合检索器 - 向量检索 + BM25关键词检索 + RRF融合 + Cross-Encoder重排序
实现多路召回策略，提升检索质量
"""
from typing import List, Optional
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import VectorStoreIndex
from rank_bm25 import BM25Okapi
import numpy as np

from app.config.settings import settings


class HybridRetriever:
    """
    混合检索器
    融合向量语义检索和BM25关键词检索，经重排序后返回最相关的文档片段
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        nodes: List[TextNode] = None,
        use_reranker: bool = True,
    ):
        """
        Args:
            index: LlamaIndex VectorStoreIndex 实例
            nodes: 所有文档节点（用于BM25索引）
            use_reranker: 是否启用重排序
        """
        self._index = index
        self._nodes = nodes or []
        self._use_reranker = use_reranker
        
        # 向量检索器
        self._vector_retriever = self._index.as_retriever(
            similarity_top_k=settings.BM25_TOP_K * 2
        )
        
        # BM25 检索器
        self._bm25 = None
        if self._nodes:
            self._build_bm25_index()
        
        # 重排序器（懒加载）
        self._reranker = None

    def _build_bm25_index(self):
        """构建BM25索引"""
        tokenized_docs = [self._tokenize(node.text) for node in self._nodes]
        self._bm25 = BM25Okapi(tokenized_docs)
        print(f"[HybridRetriever] BM25索引构建完成，共 {len(self._nodes)} 个文档")

    def _tokenize(self, text: str) -> List[str]:
        """中文分词（字符级）"""
        return list(text.replace(" ", ""))

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        similarity_threshold: float = None,
    ) -> List[NodeWithScore]:
        """
        混合检索：向量 + BM25 → RRF融合 → 重排序
        Args:
            query: 用户查询
            top_k: 返回的片段数量
            similarity_threshold: 相似度阈值
        Returns:
            带分数的节点列表
        """
        top_k = top_k or settings.TOP_K
        similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD

        # 1. 向量检索
        vector_nodes = self._vector_retriever.retrieve(query)
        print(f"[HybridRetriever] 向量检索召回 {len(vector_nodes)} 个")

        # 2. BM25 检索
        bm25_nodes = []
        if self._bm25:
            bm25_nodes = self._bm25_retrieve(query, top_k=settings.BM25_TOP_K)
            print(f"[HybridRetriever] BM25检索召回 {len(bm25_nodes)} 个")

        # 3. RRF 融合
        fused_nodes = self._reciprocal_rank_fusion(vector_nodes, bm25_nodes)
        print(f"[HybridRetriever] RRF融合后 {len(fused_nodes)} 个")

        # 4. 重排序
        if self._use_reranker and fused_nodes:
            reranked_nodes = self._rerank(query, fused_nodes, top_n=top_k)
            print(f"[HybridRetriever] 重排序后返回 {len(reranked_nodes)} 个")
        else:
            reranked_nodes = fused_nodes[:top_k]

        # 5. 过滤低于阈值的片段
        filtered_nodes = [
            node for node in reranked_nodes 
            if node.score >= similarity_threshold
        ]

        return filtered_nodes[:top_k]

    def _bm25_retrieve(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[NodeWithScore]:
        """BM25关键词检索"""
        if not self._bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # 获取 top_k 个最高分的索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                node = self._nodes[idx]
                node_with_score = NodeWithScore(
                    node=node,
                    score=float(scores[idx])
                )
                results.append(node_with_score)
        
        return results

    def _reciprocal_rank_fusion(
        self,
        *retriever_results: List[NodeWithScore],
        k: int = 60,
    ) -> List[NodeWithScore]:
        """
        RRF (Reciprocal Rank Fusion) 融合多路检索结果
        公式: RRF(d) = Σ 1 / (k + rank_i(d))
        """
        node_scores = {}
        node_map = {}

        for results in retriever_results:
            for rank, node_with_score in enumerate(results):
                node_id = node_with_score.node.node_id
                if node_id not in node_scores:
                    node_scores[node_id] = 0.0
                    node_map[node_id] = node_with_score.node
                
                # RRF 分数累加
                node_scores[node_id] += 1.0 / (k + rank + 1)

        # 按 RRF 分数排序
        sorted_ids = sorted(node_scores.keys(), key=lambda x: node_scores[x], reverse=True)
        
        fused_results = []
        for node_id in sorted_ids:
            node = node_map[node_id]
            fused_results.append(NodeWithScore(
                node=node,
                score=node_scores[node_id]
            ))

        return fused_results

    def _rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: int = 5,
    ) -> List[NodeWithScore]:
        """
        Cross-Encoder 重排序
        使用 BGE-Reranker 模型对候选文档进行精排
        """
        if not nodes:
            return []

        # 懒加载重排序器
        if self._reranker is None:
            try:
                from llama_index.core.postprocessor import SentenceTransformerReranker
                self._reranker = SentenceTransformerReranker(
                    model=settings.RERANK_MODEL,
                    top_n=top_n,
                )
                print(f"[HybridRetriever] 重排序器加载完成: {settings.RERANK_MODEL}")
            except Exception as e:
                print(f"[HybridRetriever] 重排序器加载失败，跳过重排序: {e}")
                return nodes[:top_n]

        # 使用 LlamaIndex 的 Reranker
        try:
            reranked = self._reranker.postprocess_nodes(nodes, query_str=query)
            return reranked[:top_n]
        except Exception as e:
            print(f"[HybridRetriever] 重排序失败: {e}")
            return nodes[:top_n]

    def get_context_text(self, nodes: List[NodeWithScore]) -> str:
        """
        将检索到的节点拼接为上下文文本，包含来源引用
        Args:
            nodes: 检索结果节点列表
        Returns:
            拼接后的上下文文本
        """
        context_parts = []
        for i, node in enumerate(nodes, 1):
            source = node.node.metadata.get("filename", "未知来源")
            sheet = node.node.metadata.get("sheet", "")
            row = node.node.metadata.get("row", "")
            
            # 构建来源信息
            source_info = source
            if sheet:
                source_info += f" [{sheet}表]"
            if row:
                source_info += f" 第{row}行"
            
            context_parts.append(
                f"[{i}] 来源: {source_info}\n{node.node.text}"
            )
        return "\n\n---\n\n".join(context_parts)

    def get_sources_info(self, nodes: List[NodeWithScore]) -> List[dict]:
        """
        提取来源信息（用于响应中返回给前端）
        Args:
            nodes: 检索结果节点列表
        Returns:
            来源信息列表
        """
        sources = []
        for i, node in enumerate(nodes, 1):
            sources.append({
                "index": i,
                "filename": node.node.metadata.get("filename", "未知来源"),
                "sheet": node.node.metadata.get("sheet", ""),
                "row": node.node.metadata.get("row", ""),
                "score": round(node.score, 4) if node.score else 0,
                "text_preview": node.node.text[:100] + "..." if len(node.node.text) > 100 else node.node.text,
            })
        return sources
