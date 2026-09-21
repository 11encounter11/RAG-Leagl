"""
RAG 服务 - 文档索引和检索的核心逻辑
整合混合检索器和 LangGraph Agentic RAG 工作流
"""
import uuid
from pathlib import Path
from typing import List, Optional

from app.core.rag.loader import DocumentLoader, DocumentChunker
from app.core.rag.embedder import Embedder
from app.core.rag.vector_store import VectorStoreManager
from app.core.rag.retriever import HybridRetriever
from app.core.workflow.rag_workflow import get_rag_workflow, RAGWorkflowState
from app.core.memory.db_store import db_manager


class RAGService:
    """RAG 服务 - 负责文档索引和检索"""

    def __init__(self):
        self._loader = DocumentLoader()
        self._chunker = DocumentChunker()
        self._embedder = Embedder()
        self._vector_store = VectorStoreManager(embed_model=self._embedder.embed_model)
        self._retriever: Optional[HybridRetriever] = None
        self._all_nodes = []  # 存储所有文档节点（用于BM25索引）

    def init_index(self):
        """初始化索引（服务启动时调用）"""
        try:
            self._vector_store.load_index()
            # 注意：这里暂时不传递 nodes，BM25 索引会在索引文档时构建
            self._retriever = HybridRetriever(
                self._vector_store.get_index(),
                nodes=self._all_nodes,
                use_reranker=True,
            )
            print("[RAGService] 索引初始化完成")
        except Exception as e:
            print(f"[RAGService] 索引加载失败（可能还没有数据）: {e}")

    def index_documents(self, dir_path: str) -> dict:
        """
        索引目录下的所有文档
        Args:
            dir_path: 文档目录路径
        Returns:
            索引结果信息
        """
        print(f"[RAGService] 开始索引目录: {dir_path}")

        # 1. 加载文档
        documents = self._loader.load_directory(dir_path)
        if not documents:
            return {"success": False, "message": "没有找到可加载的文档"}

        # 2. 分块
        nodes = self._chunker.chunk_documents(documents)
        self._all_nodes.extend(nodes)

        # 3. 创建向量索引
        self._vector_store.create_index(nodes)
        
        # 4. 重建混合检索器（包含BM25索引）
        self._retriever = HybridRetriever(
            self._vector_store.get_index(),
            nodes=self._all_nodes,
            use_reranker=True,
        )

        # 5. 记录到数据库
        for file_path in Path(dir_path).iterdir():
            if file_path.suffix.lower() in {".pdf", ".docx", ".xlsx", ".csv", ".md", ".txt"}:
                doc_id = str(uuid.uuid4())
                db_manager.add_document(
                    doc_id=doc_id,
                    filename=file_path.name,
                    file_type=file_path.suffix[1:],
                    file_size=file_path.stat().st_size,
                )
                db_manager.update_document_status(
                    doc_id=doc_id,
                    status="completed",
                    chunk_count=len(nodes),
                )

        return {
            "success": True,
            "document_count": len(documents),
            "chunk_count": len(nodes),
            "message": f"成功索引 {len(documents)} 个文档，共 {len(nodes)} 个片段",
        }

    def index_single_file(self, file_path: str) -> dict:
        """
        索引单个文件
        Args:
            file_path: 文件路径
        Returns:
            索引结果
        """
        print(f"[RAGService] 开始索引文件: {file_path}")

        documents = self._loader.load_file(file_path)
        nodes = self._chunker.chunk_documents(documents)
        self._all_nodes.extend(nodes)

        # 追加到已有索引
        self._vector_store.create_index(nodes)
        
        # 重建混合检索器
        self._retriever = HybridRetriever(
            self._vector_store.get_index(),
            nodes=self._all_nodes,
            use_reranker=True,
        )

        doc_id = str(uuid.uuid4())
        file = Path(file_path)
        db_manager.add_document(
            doc_id=doc_id,
            filename=file.name,
            file_type=file.suffix[1:],
            file_size=file.stat().st_size,
        )
        db_manager.update_document_status(
            doc_id=doc_id,
            status="completed",
            chunk_count=len(nodes),
        )

        return {
            "success": True,
            "document_id": doc_id,
            "chunk_count": len(nodes),
            "message": f"成功索引 {file.name}，共 {len(nodes)} 个片段",
        }

    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float = 0.5):
        """
        传统检索模式（不使用工作流）
        Args:
            query: 查询文本
            top_k: 返回片段数
            similarity_threshold: 相似度阈值
        Returns:
            (检索结果节点列表, 上下文文本)
        """
        if self._retriever is None:
            return [], ""

        nodes = self._retriever.retrieve(query, top_k, similarity_threshold)
        context = self._retriever.get_context_text(nodes)
        return nodes, context

    async def retrieve_with_workflow(
        self, 
        query: str, 
        top_k: int = 5, 
        similarity_threshold: float = 0.5
    ) -> dict:
        """
        Agentic RAG 工作流模式
        Args:
            query: 查询文本
            top_k: 返回片段数
            similarity_threshold: 相似度阈值
        Returns:
            工作流执行结果
        """
        if self._retriever is None:
            return {
                "context": "",
                "sources": [],
                "workflow_steps": [],
                "reflection_score": 0.0,
            }

        # 获取工作流实例
        workflow = get_rag_workflow(self._retriever)

        # 初始化状态
        initial_state: RAGWorkflowState = {
            "query": query,
            "rewritten_query": "",
            "hyde_doc": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "context": "",
            "reflection_score": 0.0,
            "answer": "",
            "sources": [],
            "retry_count": 0,
            "workflow_steps": [],
        }

        # 执行工作流
        result = await workflow.ainvoke(initial_state)

        return {
            "context": result.get("context", ""),
            "sources": result.get("sources", []),
            "workflow_steps": result.get("workflow_steps", []),
            "reflection_score": result.get("reflection_score", 0.0),
        }

    def get_sources_info(self, nodes):
        """获取来源信息"""
        if self._retriever is None:
            return []
        return self._retriever.get_sources_info(nodes)


# 全局实例
rag_service = RAGService()
