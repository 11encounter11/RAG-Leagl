"""
知识库路由 - 文档上传与管理

改动要点:
    1. 上传支持 `doc_type` 参数(用户显式指定文档类型是切分策略的主路径,
       详见 docs/02a-文档解析与切分策略设计.md);
    2. 落库保留**原始文件名**(原来只存 uuid 文件名, 界面上无法辨认);
    3. 返回真实的 doc_type / 切分策略 / 是否降级, 让用户看得见处理结果;
    4. 新增删除接口 —— 删除必须同时清理向量, 否则会留下孤儿向量(F1.9);
    5. 新增片段回查接口, 支撑"引用可核验"这条硬需求。
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.memory.db_store import db_manager
from app.models.schemas import DocumentInfo, DocumentUploadResponse
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["知识库"])

UPLOAD_DIR = "data/raw"

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".md", ".txt"}

# 与 app/core/ingest/registry.py 中注册的类型保持一致
SUPPORTED_DOC_TYPES = {"statute", "faq", "generic"}


@router.post("/upload", response_model=DocumentUploadResponse, summary="上传文档")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: Optional[str] = Form(
        None, description="文档类型: statute(法律法规) / faq(问答) / generic, 不传则自动判定"
    ),
    kb_id: str = Form("default"),
):
    """上传文档并索引到向量数据库。"""
    original_filename = file.filename or "unknown"
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if doc_type and doc_type not in SUPPORTED_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"未知的 doc_type: {doc_type}，可选: {', '.join(sorted(SUPPORTED_DOC_TYPES))}",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")

    with open(saved_path, "wb") as handle:
        handle.write(await file.read())

    try:
        result = rag_service.index_file(
            saved_path,
            original_filename=original_filename,
            doc_type=doc_type,
            kb_id=kb_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("文档索引失败: %s", original_filename)
        raise HTTPException(status_code=500, detail=f"文档索引失败: {exc}") from exc

    if not result.get("success"):
        # 重复文档不是错误, 用 409 明确区分开
        if result.get("duplicate"):
            raise HTTPException(status_code=409, detail=result["message"])
        raise HTTPException(status_code=400, detail=result.get("message", "索引失败"))

    return DocumentUploadResponse(
        document_id=result["document_id"],
        filename=original_filename,
        status="completed",
        chunk_count=result.get("chunk_count", 0),
        doc_type=result.get("doc_type"),
        chunk_strategy=result.get("chunk_strategy"),
        degraded=bool(result.get("degraded")),
        message=result["message"],
    )


@router.get("/documents", summary="获取文档列表")
async def list_documents(
    limit: int = 20,
    offset: int = 0,
    kb_id: Optional[str] = None,
):
    """获取已索引的文档列表。"""
    docs = db_manager.list_documents(limit=limit, offset=offset, kb_id=kb_id)
    return {
        "total": db_manager.count_documents(kb_id=kb_id),
        "items": [
            DocumentInfo(
                id=d.id,
                filename=d.filename,
                original_filename=d.original_filename,
                file_type=d.file_type,
                file_size=d.file_size,
                chunk_count=d.chunk_count,
                status=d.status,
                kb_id=d.kb_id,
                doc_type=d.doc_type,
                doc_type_source=d.doc_type_source,
                chunk_strategy=d.chunk_strategy,
                strategy_confidence=d.strategy_confidence,
                structure_detected=d.structure_detected,
                created_at=d.created_at,
                processed_at=d.processed_at,
                error_message=d.error_message,
            )
            for d in docs
        ],
    }


@router.delete("/documents/{document_id}", summary="删除文档")
async def delete_document(document_id: str):
    """删除文档: 记录与向量一起清理。"""
    document = db_manager.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"文档不存在: {document_id}")
    return rag_service.delete_document(document_id)


@router.get("/chunks/{chunk_id}", summary="回查片段原文")
async def get_chunk(chunk_id: str):
    """按片段 ID 取回原文与元数据 —— 引用核验的落点。"""
    record = rag_service._vector_store.get_chunk(chunk_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"片段不存在: {chunk_id}")
    return {
        "chunk_id": record["id"],
        "text": record["text"],
        "metadata": record.get("metadata") or {},
    }


@router.post("/index-dir", summary="索引目录")
async def index_directory(
    dir_path: str = "data/raw",
    doc_type: Optional[str] = None,
    kb_id: str = "default",
):
    """索引指定目录下的所有文档。"""
    if not os.path.exists(dir_path):
        raise HTTPException(status_code=404, detail=f"目录不存在: {dir_path}")

    result = rag_service.index_directory(
        dir_path, doc_type=doc_type, kb_id=kb_id
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result
