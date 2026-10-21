"""
RAG 服务 - 文档索引与检索的编排层

职责边界:
    本模块只做"编排": 取文件 -> 交给 ingest 模块解析 -> 入库 -> 建检索器。
    解析与切分策略在 app/core/ingest/, 检索实现在 app/core/rag/retriever.py。
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config.settings import settings
from app.core.ingest import build_text_nodes, ingest_text
from app.core.law_versions import load_version_ref_index, make_article_id
from app.core.memory.db_store import db_manager
from app.core.rag.embedder import Embedder
from app.core.rag.loader import DocumentLoader
from app.core.rag.retriever import HybridRetriever
from app.core.rag.vector_store import VectorStoreManager
from app.core.workflow.rag_workflow import RAGWorkflowState, get_rag_workflow, params_for_round

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".md", ".txt"}


def compute_content_hash(path: str | Path) -> str:
    """文件内容哈希, 用于重复上传去重(F1.8)。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RAGService:
    """RAG 服务 - 文档索引和检索"""

    def __init__(self) -> None:
        self._loader = DocumentLoader()
        self._embedder = Embedder()
        self._vector_store = VectorStoreManager(embed_model=self._embedder.embed_model)
        self._retriever: Optional[HybridRetriever] = None
        # 文档级元数据默认值。单用户自用, 知识库分组默认 default。
        self.default_kb_id = "default"
        # 语料清单的"文件名 → 版本"索引（懒加载）。入库时按文件名决定
        # law_articles.version_id，见 T-25。
        self._version_refs: Optional[Dict[str, Any]] = None

    # ============================================================
    # 生命周期
    # ============================================================
    def init_index(self, *, rebuild_bm25: bool = True, preload_reranker: bool = True) -> None:
        """初始化索引(服务启动时调用)。

        修复点:
          T-10 原实现传入的 nodes 恒为空列表, 导致 BM25 语料永久为空 ——
               重启后关键词召回整路失效, 而且不报错, 只是"检索变差了"。
               现在启动期从 Milvus 读回正文重建语料。
          T-11 重排器改为启动期预热, 不再让第一个用户等模型加载。
        """
        try:
            self._vector_store.load_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RAGService] 索引加载失败(可能还没有数据): %s", exc)

        index = self._vector_store.get_index()
        if index is None:
            logger.info("[RAGService] 尚无索引, 等待首次上传文档")
            return

        self._retriever = HybridRetriever(index, nodes=[], use_reranker=settings.ENABLE_RERANK)

        if rebuild_bm25:
            count = self.rebuild_bm25_corpus()
            logger.info("[RAGService] BM25 语料重建完成, 共 %d 条", count)
        if preload_reranker:
            ready = self._retriever.preload_reranker()
            logger.info("[RAGService] 重排器预热: %s", "就绪" if ready else "未启用/失败")

    def rebuild_bm25_corpus(self) -> int:
        """从向量库读回正文, 重建 BM25 语料。"""
        if self._retriever is None:
            return 0
        try:
            records = self._vector_store.iter_chunks()
        except Exception as exc:  # noqa: BLE001
            logger.error("[RAGService] 读取向量库失败, BM25 语料为空: %s", exc)
            return 0
        if not records:
            logger.warning("[RAGService] 向量库中没有可读正文, BM25 将不可用")
            return 0
        return self._retriever.set_bm25_corpus_from_records(records)

    def _rebuild_retriever(self) -> None:
        """索引内容变化后重建检索器。

        注意: 这里换的是 self._retriever。工作流不再持有 retriever(见 T-07),
        所以换实例之后新文档立刻可被 agentic 模式检索到。
        """
        index = self._vector_store.get_index()
        if index is None:
            return
        nodes: List[Any] = []
        try:
            nodes = self._vector_store.rebuild_bm25_nodes()
        except Exception as exc:  # noqa: BLE001
            logger.error("[RAGService] 重建 BM25 语料失败: %s", exc)
        self._retriever = HybridRetriever(index, nodes=nodes, use_reranker=settings.ENABLE_RERANK)

    # ============================================================
    # 索引
    # ============================================================
    def _extract_text(self, file_path: str) -> str:
        """把任意支持格式的文件提取为纯文本(交给 ingest 模块处理结构)。"""
        documents = self._loader.load_file(file_path)
        return "\n\n".join((doc.text or "") for doc in documents).strip()

    # ============================================================
    # 条文层（T-25）
    # ============================================================
    def _version_index(self) -> Dict[str, Any]:
        """语料清单的"文件名 → 版本标识"索引（只加载一次）。

        入库时用它决定 `law_articles.version_id`。**按文件名查，不查数据库** ——
        因为入库那一刻 `documents` 还没写、清库重入时库里更是空的，
        而文件名在入库前后都稳定（理由详见 `app/core/law_versions.py`）。
        """
        if self._version_refs is None:
            self._version_refs = load_version_ref_index()
            if not self._version_refs:
                logger.warning(
                    "[RAGService] 语料清单缺失或为空（data/corpus_manifest.json）—— "
                    "条文仍会入库，但 version_id 为空，时效过滤（T-26~T-28）将失效"
                )
        return self._version_refs

    def _document_meta(
        self,
        *,
        original_filename: str,
        kb_id: str,
        document_id: str,
        source_file: str,
    ) -> Dict[str, Any]:
        """组装**文档级**元数据，含真实的版本与时效（T-26）。

        在这之前，chunk 里的 `validity_status` 恒为占位值 `effective` ——
        后果是同一部法的两个版本（如劳动法 2009 / 2018）**都被标成"现行有效"**，
        检索会把它们一起召回，而且看不出哪个是现行的（实测证据见 04-P1 T-25 完成记录）。

        版本信息按**文件名**从语料清单取 —— 与 `_write_law_articles` 用的是同一份
        索引，所以 chunk 元数据里的 `version_id` 与 `law_articles.version_id`
        **必然一致**，不会出现"两边各说各话"。

        `effective_date` 存成 ISO 字符串（`YYYY-MM-DD`）：Milvus 该字段声明为 VARCHAR，
        而 ISO 格式恰好可以按字典序比较，范围过滤仍然成立。
        """
        meta: Dict[str, Any] = {
            "kb_id": kb_id,
            "document_id": document_id,
            "source_file": source_file,
            "filename": original_filename,
        }
        ref = self._version_index().get(original_filename)
        if ref is not None:
            meta.update(
                {
                    "version_id": ref.version_id,
                    "version_label": ref.version_label,
                    "validity_status": ref.validity_status,
                    "effective_date": (
                        ref.effective_date.isoformat() if ref.effective_date else None
                    ),
                }
            )
        return meta

    def _write_law_articles(self, result, original_filename: str) -> int:
        """把解析出的条文写进 `law_articles`（权威源）。返回写入条数。

        `version_id` 为空不算致命错误 —— 只说明这份文件不在语料清单里
        （例如用户临时上传的文档），此时条文原文仍然值得留存。
        """
        ref = self._version_index().get(original_filename)
        if ref is None:
            logger.warning(
                "[RAGService] %s 不在语料清单中 → 条文将以空 version_id 入库，"
                "该文档不会参与时效过滤",
                original_filename,
            )
        version_id = ref.version_id if ref else ""

        rows = [
            {
                "id": make_article_id(version_id or None, section.article_no, section.seq),
                "version_id": version_id,
                "article_no": section.article_no or "",
                "article_no_norm": section.article_no_norm,
                "chapter_path": section.chapter_path,
                "seq": section.seq,
                "content": section.text,
            }
            for section in result.sections
        ]

        # 若版本种子没灌过，version_id 会悬空 → 时效过滤静默失效。这里留痕。
        if version_id and not db_manager.law_version_exists(version_id):
            logger.warning(
                "[RAGService] law_versions 里没有 %s —— 请先跑 "
                "scripts/seed_law_versions.py，否则这条 doc 的时效过滤不生效",
                version_id,
            )
        return db_manager.upsert_law_articles(rows)

    def index_file(
        self,
        file_path: str,
        *,
        original_filename: Optional[str] = None,
        doc_type: Optional[str] = None,
        kb_id: Optional[str] = None,
        doc_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """索引单个文件。

        Returns:
            {"success", "document_id", "chunk_count", "doc_type", "chunk_strategy",
             "strategy_confidence", "structure_detected", "message"}
        """
        path = Path(file_path)
        kb_id = kb_id or self.default_kb_id
        original_filename = original_filename or path.name
        content_hash = compute_content_hash(path)

        # ---- 去重(F1.8): 同一知识库内内容相同就不再入库 ----
        existing = db_manager.find_document_by_hash(kb_id, content_hash)
        if existing:
            logger.info("[RAGService] 内容重复, 跳过入库: %s", original_filename)
            return {
                "success": False,
                "duplicate": True,
                "document_id": existing.id,
                "chunk_count": existing.chunk_count,
                "message": f"该文档内容已存在（{existing.original_filename or existing.filename}），未重复入库",
            }

        text = self._extract_text(str(path))
        if not text:
            # 需求 F1.2: PDF 要区分"有文字层"和"扫描件"。
            # OCR 已明确不做(见 01-需求文档.md §四), 所以扫描件在这里被明确拒绝并说明原因,
            # 而不是产出一个空的 chunk 列表(那会让"扫描件"伪装成"文档里没有内容")。
            if path.suffix.lower() == ".pdf":
                return {
                    "success": False,
                    "message": "该 PDF 未提取到文字层，可能是扫描件；当前版本不支持 OCR，"
                    "请上传有文字层的 PDF 或纯文本文件。",
                }
            return {"success": False, "message": "文件内容为空或无法解析"}

        document_id = str(uuid.uuid4())
        meta: Dict[str, Any] = self._document_meta(
            original_filename=original_filename,
            kb_id=kb_id,
            document_id=document_id,
            source_file=str(path),
        )
        meta.update(doc_meta or {})

        result, type_source, route_detail = ingest_text(
            text,
            filename=original_filename,
            user_hint=doc_type,
            doc_meta=meta,
        )

        if not result.sections:
            db_manager.add_document(
                doc_id=document_id,
                filename=path.name,
                original_filename=original_filename,
                file_type=path.suffix[1:].lower(),
                file_size=path.stat().st_size,
                kb_id=kb_id,
                content_hash=content_hash,
                doc_type=result.doc_type,
                doc_type_source=type_source,
                chunk_strategy=result.strategy,
                strategy_confidence=result.confidence,
                structure_detected=result.structure_detected,
            )
            db_manager.update_document_status(
                document_id, status="failed", error_message="解析后没有任何可用片段"
            )
            return {"success": False, "document_id": document_id, "message": "解析后没有任何可用片段"}

        nodes = build_text_nodes(result, doc_meta=meta)
        self._vector_store.create_index(nodes)  # 追加写入
        self._rebuild_retriever()

        db_manager.add_document(
            doc_id=document_id,
            filename=path.name,
            original_filename=original_filename,
            file_type=path.suffix[1:].lower(),
            file_size=path.stat().st_size,
            kb_id=kb_id,
            content_hash=content_hash,
            doc_type=result.doc_type,
            doc_type_source=type_source,
            chunk_strategy=result.strategy,
            strategy_confidence=result.confidence,
            structure_detected=result.structure_detected,
        )
        db_manager.update_document_status(
            document_id, status="completed", chunk_count=len(nodes)
        )

        # ---- 条文层（T-25）----
        # `law_articles` 是条文的**权威源**，向量库里的 chunk 只是它的检索索引。
        # 放在 documents 之后写：两者同属 MySQL 侧，失败时至少文档级记录是完整的。
        article_count = self._write_law_articles(result, original_filename)

        logger.info(
            "[RAGService] 索引完成: %s -> %d 片段 / %d 条文 | doc_type=%s strategy=%s confidence=%s source=%s",
            original_filename,
            len(nodes),
            article_count,
            result.doc_type,
            result.strategy,
            result.confidence,
            type_source,
        )

        return {
            "success": True,
            "document_id": document_id,
            "chunk_count": len(nodes),
            "doc_type": result.doc_type,
            "doc_type_source": type_source,
            "chunk_strategy": result.strategy,
            "strategy_confidence": result.confidence,
            "structure_detected": result.structure_detected,
            "route_detail": route_detail,
            "degraded": result.is_degraded(),
            "message": f"成功索引 {original_filename}，共 {len(nodes)} 个片段"
            + ("（结构未识别，已降级处理）" if result.is_degraded() else ""),
        }

    def index_directory(self, dir_path: str, **kwargs: Any) -> Dict[str, Any]:
        """索引目录下的所有文档。每个文件独立统计, 不再共用总块数。"""
        directory = Path(dir_path)
        if not directory.is_dir():
            return {"success": False, "message": f"目录不存在: {dir_path}"}

        results: List[Dict[str, Any]] = []
        for file_path in sorted(directory.iterdir()):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                results.append(self.index_file(str(file_path), **kwargs))
            except Exception as exc:  # noqa: BLE001 - 单个文件失败不应中断整批
                logger.error("[RAGService] 索引失败 %s: %s", file_path.name, exc, exc_info=True)
                results.append({"success": False, "message": f"{file_path.name}: {exc}"})

        succeeded = [r for r in results if r.get("success")]
        duplicates = [r for r in results if r.get("duplicate")]
        return {
            "success": bool(succeeded),
            "document_count": len(succeeded),
            "duplicate_count": len(duplicates),
            "chunk_count": sum(int(r.get("chunk_count") or 0) for r in succeeded),
            "documents": results,
            "message": (
                f"成功索引 {len(succeeded)} 个文档, "
                f"共 {sum(int(r.get('chunk_count') or 0) for r in succeeded)} 个片段"
                + (f", 跳过重复 {len(duplicates)} 个" if duplicates else "")
            ),
        }

    # 兼容旧调用名
    def index_single_file(self, file_path: str, original_filename: Optional[str] = None) -> Dict[str, Any]:
        return self.index_file(file_path, original_filename=original_filename)

    def index_documents(self, dir_path: str) -> Dict[str, Any]:
        return self.index_directory(dir_path)

    # ============================================================
    # 删除(F1.9: 必须同时清理向量, 否则留下永远删不掉的孤儿向量)
    # ============================================================
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        removed_vectors = self._vector_store.delete_by_document(document_id)
        db_manager.delete_document(document_id)
        self._rebuild_retriever()
        logger.info(
            "[RAGService] 文档 %s 已删除, 清理向量 %d 条", document_id, removed_vectors
        )
        return {
            "success": True,
            "document_id": document_id,
            "removed_vectors": removed_vectors,
            "message": f"文档已删除，同时清理了 {removed_vectors} 条向量",
        }

    # ============================================================
    # 检索
    # ============================================================
    def retrieve(
        self, query: str, top_k: Optional[int] = None, similarity_threshold: Optional[float] = None
    ) -> Tuple[List[Any], str]:
        """传统检索(不走工作流)。"""
        if self._retriever is None:
            return [], ""
        nodes = self._retriever.retrieve(query, top_k, similarity_threshold)
        return nodes, self._retriever.get_context_text(nodes)

    @property
    def last_threshold_info(self) -> Dict[str, Any]:
        """最近一次直连检索的阈值闸门实况, 供留痕。工作流走自己的状态, 不读这里。"""
        if self._retriever is None:
            return {}
        return self._retriever.last_threshold_info

    async def retrieve_with_workflow(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Agentic RAG 工作流模式。

        修复 T-03: 返回值里补上 reranked_docs。
        原实现只返回 context/sources/workflow_steps/reflection_score,
        而 chat_service 却在取 `rag_result.get("reranked_docs", [])` ——
        取到的永远是空列表, 导致响应里的引用条数恒为 0。
        """
        if self._retriever is None:
            return {
                "context": "",
                "sources": [],
                "reranked_docs": [],
                "workflow_steps": ["retriever_not_ready"],
                "reflection_score": 0.0,
                "reflection_core_basis": None,
                "reflection_coverage": "",
                "reflection_missing_type": "",
                "reflection_missing": [],
                "reflection_missing_query": "",
                "refuse": True,
            }

        initial_params = params_for_round(0)
        if top_k is not None:
            initial_params["top_k"] = top_k
        if similarity_threshold is not None:
            initial_params["threshold"] = similarity_threshold

        initial_state: RAGWorkflowState = {
            "query": query,
            "rewritten_query": "",
            "law_name": "",
            "hyde_doc": "",
            "sub_questions": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "context": "",
            "reflection_score": 0.0,
            "reflection_note": "",
            "reflection_core_basis": None,
            "reflection_coverage": "",
            "reflection_missing_type": "",
            "reflection_missing": [],
            "reflection_missing_query": "",
            "sources": [],
            "retry_count": 0,
            "retry_params": initial_params,
            "refuse": False,
            "domain_score": 1.0,
            "confidence_level": "high",
            "refuse_reason": "",
            "threshold_usage": {},
            "workflow_steps": [],
        }

        workflow = get_rag_workflow()
        # retriever 按请求注入 —— 这是 T-07 的修复点, 不再有全局单例绑定旧索引
        result = await workflow.ainvoke(
            initial_state, config={"configurable": {"retriever": self._retriever}}
        )

        return {
            "context": result.get("context", ""),
            "sources": result.get("sources", []),
            "reranked_docs": result.get("reranked_docs", []),
            # 改写后的那句要能传出去: 它决定了本次检索实际用的输入。
            # 以前只在工作流状态里, 外面拿不到 —— 出问题只能自己包一层 LLM 去看。
            "rewritten_query": result.get("rewritten_query", ""),
            "workflow_steps": result.get("workflow_steps", []),
            "reflection_score": result.get("reflection_score", 0.0),
            "reflection_note": result.get("reflection_note", ""),
            # P2-04: 结构化缺口。core_basis/coverage 决定路由, missing 进生成提示。
            "reflection_core_basis": result.get("reflection_core_basis"),
            "reflection_coverage": result.get("reflection_coverage", ""),
            "reflection_missing_type": result.get("reflection_missing_type", ""),
            "reflection_missing": list(result.get("reflection_missing") or []),
            "reflection_missing_query": result.get("reflection_missing_query", ""),
            "refuse": bool(result.get("refuse", False)),
            "refuse_reason": result.get("refuse_reason", ""),
            "domain_score": result.get("domain_score", 0.0),
            "confidence_level": result.get("confidence_level", "high"),
            "retry_count": result.get("retry_count", 0),
            # 阈值闸门的尺子名/值/top1/保留条数 —— 写进 retrieval_info, 事后可复盘
            "threshold_usage": result.get("threshold_usage") or {},
        }

    def get_sources_info(self, nodes: List[Any]) -> List[dict]:
        if self._retriever is None or not nodes:
            return []
        return self._retriever.get_sources_info(nodes)


# 全局实例
rag_service = RAGService()
