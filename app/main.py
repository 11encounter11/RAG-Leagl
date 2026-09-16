"""
FastAPI 应用入口
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.core.memory.db_store import db_manager
from app.core.memory.redis_store import redis_store
from app.services.rag_service import rag_service
from app.core.llm.router import llm_router
from app.routers import chat, knowledge, conversation
from app.models.schemas import HealthStatus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("=" * 50)
    print("法律智能问答系统 - 启动中...")
    print("=" * 50)

    # 初始化数据库
    db_manager.init_db()

    # 初始化 RAG 索引
    rag_service.init_index()

    print(f"可用模型: {llm_router.list_models()}")
    print("系统启动完成!")
    print("=" * 50)

    yield

    # 关闭时
    print("系统关闭中...")


# 创建 FastAPI 应用
app = FastAPI(
    title="法律智能问答系统",
    description="基于 RAG 的法律智能问答系统，支持多模型切换、流式输出、多轮对话",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(conversation.router)


@app.get("/api/health", response_model=HealthStatus, tags=["系统"])
async def health_check():
    """健康检查接口"""
    mysql_ok = db_manager.ping()
    redis_ok = await redis_store.ping()

    # 检查 Milvus
    milvus_ok = False
    try:
        from pymilvus import connections
        connections.connect(uri=settings.milvus_uri)
        milvus_ok = connections.has_connection("default")
        connections.disconnect("default")
    except Exception:
        pass

    return HealthStatus(
        status="ok" if (mysql_ok and redis_ok and milvus_ok) else "degraded",
        mysql=mysql_ok,
        redis=redis_ok,
        milvus=milvus_ok,
    )


@app.get("/api/models", tags=["系统"])
async def list_models():
    """获取可用模型列表"""
    return {"models": llm_router.list_models()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
