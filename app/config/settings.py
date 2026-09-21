"""
应用配置管理
从 .env 文件读取配置，提供全局配置对象
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置，从环境变量或 .env 文件加载"""

    # ===== LLM API Keys =====
    ZHIPU_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""

    # ===== MySQL =====
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3307
    MYSQL_USER: str = "rag_user"
    MYSQL_PASSWORD: str = "rag_password"
    MYSQL_DATABASE: str = "rag_system"

    # ===== Redis =====
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0

    # ===== Milvus =====
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # ===== RAG Settings =====
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.5
    BM25_TOP_K: int = 10
    RERANK_TOP_N: int = 5
    # 重排器一次性最多精排多少条候选。
    # 取大值是刻意的: 重排器创建时就要定死 top_n, 而我们希望拿到全部候选的
    # 原始分再做归一化与排序, 所以给它一个足够大的上限, 由调用方自行截断。
    RERANK_CANDIDATE_LIMIT: int = 50
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    # 重排器输出的是不是原始 logits?
    #   False(默认): 认为 Cross-Encoder 已按模型 config 套过 Sigmoid, 输出即 [0,1] 概率
    #   True       : 上游返回原始 logits, 由我们施加 sigmoid
    # 这个开关必须显式指定, 不能靠"值落在哪判断" —— 那样在 0 处不连续。
    # 若实际返回越界值, 运行时会打印 ERROR 日志提示改这里。
    RERANK_SCORE_IS_LOGIT: bool = False

    # ===== Agentic RAG Settings =====
    REFLECTION_THRESHOLD: float = 0.6
    MAX_RETRIES: int = 2
    QUERY_REWRITE_MODEL: str = "glm-5.3-flash"
    HYDE_MODEL: str = "glm-5.3-flash"
    # 反思节点专用的模型。刻意与改写模型分开配置 —— 用同一个模型评估自己
    # 的检索结果属于自评自夸, 至少要让这件事是显式可调的。
    REFLECTION_MODEL: str = "glm-5.3-flash"

    # ===== 功能开关 =====
    # P0 阶段先放在配置里(服务端生效、可验证); P3 迁移到数据库后台管理。
    # 关掉某个开关后, 工作流会真的不执行那个节点 —— 不是"前端不传参"那种伪生效。
    ENABLE_AGENTIC_RAG: bool = True
    ENABLE_QUERY_REWRITE: bool = True
    ENABLE_HYDE: bool = False          # 默认关: 成本高, 收益尚未被评测验证
    ENABLE_RERANK: bool = True
    ENABLE_CITATION_VERIFY: bool = True

    # ===== Embedding Model =====
    EMBEDDING_MODEL_NAME: str = "embedding-3"
    # 智谱 embedding-3 的向量维度。向量库建集合时要用, 所以必须可配 ——
    # 写死在 vector_store.py 里的话, 换模型就得改代码。
    EMBEDDING_DIM: int = 2048

    # ===== Server =====
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # ===== Milvus Collection =====
    MILVUS_COLLECTION_NAME: str = "knowledge_base"

    @property
    def mysql_url(self) -> str:
        """MySQL 连接 URL"""
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

    @property
    def redis_url(self) -> str:
        """Redis 连接 URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def milvus_uri(self) -> str:
        """Milvus 连接 URI"""
        return f"http://{self.MILVUS_HOST}:{self.MILVUS_PORT}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()
