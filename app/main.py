"""
FastAPI 应用入口
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 支持 `python app/main.py` 直接启动。
# 直接跑脚本时 sys.path[0] 是 app/ 而不是项目根, 于是 `import app.xxx` 会报
# ModuleNotFoundError: No module named 'app'。scripts/ 下的脚本都有这一行,
# 入口脚本同样需要 —— 否则唯一可用的启动方式是 `python -m app.main`,
# 而这正是"照着文档做却起不来"的典型来源。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.core.memory.db_store import db_manager  # noqa: E402
from app.core.memory.redis_store import redis_store  # noqa: E402
from app.services.rag_service import rag_service  # noqa: E402
from app.core.llm.router import llm_router  # noqa: E402
from app.routers import chat, knowledge, conversation  # noqa: E402
from app.models.schemas import HealthStatus  # noqa: E402

# 统一日志配置。原来全靠 print, 出问题时既没有级别也没有模块名,
# 排查只能靠猜 —— 这是 P0 里最便宜也最有价值的一处改动。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("个人法律知识问答助手 - 启动中")
    logger.info("=" * 60)

    db_manager.init_db()

    # 初始化 RAG 索引: 启动期完成 BM25 语料重建与重排器预热(T-10 / T-11)。
    # 这两件事原来都发生在"第一次请求"时, 前者导致关键词召回静默失效,
    # 后者让第一个用户白等一两个 G 的模型加载。
    rag_service.init_index()

    logger.info("可用模型: %s", llm_router.list_models())
    logger.info("BM25 语料: %s 条", rag_service._retriever.bm25_size if rag_service._retriever else 0)
    logger.info("系统启动完成")
    logger.info("=" * 60)

    yield

    logger.info("系统关闭中...")


# 创建 FastAPI 应用
app = FastAPI(
    title="个人法律知识问答助手",
    description="个人自用的法律知识问答助手：按条切分、引用可核验、时效可标注、查不到就拒答",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件。来源走配置(默认仅本机前端), 不用 "*" + allow_credentials=True
# 的组合 —— 浏览器会直接拒绝该组合, 且等价于完全开放(见 02 §13 P1 #20)。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
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
