"""
知识库路由 - 文档上传和管理
"""
import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import DocumentInfo, DocumentUploadResponse
from app.core.memory.db_store import db_manager
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/knowledge", tags=["知识库"])

UPLOAD_DIR = "data/raw"


@router.post("/upload", response_model=DocumentUploadResponse, summary="上传文档")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并索引到向量数据库
    支持: PDF, DOCX, MD, TXT
    """
    # 验证文件类型
    allowed_types = {".pdf", ".docx", ".md", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {', '.join(allowed_types)}",
        )

    # 保存文件
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_id = str(uuid.uuid4())
    saved_filename = f"{file_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 索引文档
    try:
        result = rag_service.index_single_file(file_path)
        return DocumentUploadResponse(
            document_id=result["document_id"],
            filename=file.filename,
            status="completed",
            message=result["message"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档索引失败: {str(e)}")


@router.get("/documents", summary="获取文档列表")
async def list_documents(limit: int = 20, offset: int = 0):
    """获取已索引的文档列表"""
    docs = db_manager.list_documents(limit=limit, offset=offset)
    return [
        DocumentInfo(
            id=d.id,
            filename=d.filename,
            file_type=d.file_type,
            file_size=d.file_size,
            chunk_count=d.chunk_count,
            status=d.status,
            created_at=d.created_at,
            processed_at=d.processed_at,
            error_message=d.error_message,
        )
        for d in docs
    ]


@router.post("/index-dir", summary="索引目录")
async def index_directory(dir_path: str = "data/raw"):
    """索引指定目录下的所有文档"""
    if not os.path.exists(dir_path):
        raise HTTPException(status_code=404, detail=f"目录不存在: {dir_path}")

    result = rag_service.index_documents(dir_path)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    return result
