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
    RERANK_MODEL: str = "BAAI/bge-reranker-base"

    # ===== Agentic RAG Settings =====
    REFLECTION_THRESHOLD: float = 0.6
    MAX_RETRIES: int = 2
    QUERY_REWRITE_MODEL: str = "glm-5.3-flash"
    HYDE_MODEL: str = "glm-5.3-flash"

    # ===== Embedding Model =====
    EMBEDDING_MODEL_NAME: str = "embedding-3"

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
