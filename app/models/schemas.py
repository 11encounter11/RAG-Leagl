"""
Pydantic 数据模型 - 请求/响应 Schema
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ===== 模型选择 =====
class ModelName(str, Enum):
    """支持的模型列表"""
    GLM_FLASH = "glm-5.3-flash"
    DEEPSEEK_FLASH = "deepseek-v4.1-flash"


# ===== 聊天相关 =====
class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., min_length=1, description="用户消息")
    model_name: ModelName = Field(ModelName.GLM_FLASH, description="选择的模型")
    conversation_id: Optional[str] = Field(None, description="会话ID，不传则创建新会话")
    top_k: Optional[int] = Field(5, description="检索返回的文档片段数")
    similarity_threshold: Optional[float] = Field(0.5, description="相似度阈值")
    use_agentic_rag: Optional[bool] = Field(True, description="是否启用 Agentic RAG 工作流")


class SourceInfo(BaseModel):
    """来源引用信息"""
    index: int
    filename: str
    sheet: Optional[str] = ""
    row: Optional[str] = ""
    score: float = 0.0
    text_preview: str = ""


class ChatResponse(BaseModel):
    """聊天响应（非流式）"""
    conversation_id: str
    message_id: int
    content: str
    model_name: str
    retrieval_count: int = 0
    tokens_used: int = 0
    response_time: float = 0.0
    sources: List[SourceInfo] = []
    workflow_steps: List[str] = []
    reflection_score: Optional[float] = None


class SSEChatResponse(BaseModel):
    """SSE 流式响应的单个 token 片段"""
    type: str = "token"  # token / done / error / sources
    content: str = ""
    conversation_id: str = ""
    model_name: str = ""
    retrieval_count: int = 0
    sources: Optional[List[SourceInfo]] = None


# ===== 会话相关 =====
class ConversationCreate(BaseModel):
    """创建会话"""
    title: Optional[str] = Field("新对话", description="会话标题")
    model_name: ModelName = Field(ModelName.GLM_FLASH, description="默认模型")


class ConversationInfo(BaseModel):
    """会话信息"""
    id: str
    title: str
    model_name: str
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class MessageInfo(BaseModel):
    """消息信息"""
    id: int
    role: str
    content: str
    model_name: Optional[str] = None
    created_at: datetime


# ===== 知识库相关 =====
class DocumentInfo(BaseModel):
    """文档信息"""
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    created_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    document_id: str
    filename: str
    status: str
    message: str


# ===== 健康检查 =====
class HealthStatus(BaseModel):
    """服务健康状态"""
    status: str = "ok"
    mysql: bool = False
    redis: bool = False
    milvus: bool = False
