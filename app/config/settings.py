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
    # 加载精度: float32(默认, 精度最高但内存占用最大) / float16 / bfloat16。
    # 为什么需要这个旋钮: bge-reranker-base 以 float32 加载约需 1GB 常驻内存,
    # 曾实测导致开发机 OOM(N-9, 见 docs/06-P0测试执行报告.md §3)。float16/bfloat16
    # 直接把权重内存占用砍半, 是"内存吃紧但不想换模型"时的第一档缓解手段。
    # 注意: CPU 上 float16 的部分算子支持有限, 若报错可改用 bfloat16 或退回 float32。
    RERANK_TORCH_DTYPE: str = "float32"
    # 推理后端: torch(默认) / onnx / openvino。
    # onnx/openvino 后端配合量化模型可以显著降低内存与提升 CPU 推理速度,
    # 但需要模型本身已导出为对应格式(或安装 optimum 现场导出), 不是换个字符串就生效。
    RERANK_BACKEND: str = "torch"

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

    # ===== 拒答闸门(独立判据) =====
    # 为什么必须独立于重排器: 拒答是法律场景的安全底线, 而重排器是可选的、
    # 可能在环境里加载不起来的组件。把底线建立在可选组件之上, 结果是
    # "重排器加载失败 -> 该拒答的全都不拒答"(实测拒答准确率 0%)。
    #
    # 判据: 向量检索的**最高原始相似度**。它与知识库主题的相关程度直接对应,
    # 不经过 RRF 融合、不依赖重排器。
    #
    # 阈值是标定出来的, 不是拍的: 用 scripts/calibrate_domain_gate.py 在评测集上
    # 跑出分布, 再选点。当前值来自 36 条库内问题 + 4 条超纲问题的实测分布
    # (库内 min=0.259 / 超纲 max=0.275), **样本量很小, 扩大评测集后必须重标**。
    ENABLE_DOMAIN_GATE: bool = True
    # 低于此值: 判定为"知识库外", 直接拒答
    DOMAIN_SCORE_REJECT: float = 0.24
    # 低于此值(但不低于 REJECT): 照常回答, 但标记为低置信, 提示用户核对
    # 采用两档而不是单阈值, 是因为实测下两个分布**区间重叠**:
    # 任何单阈值都必然误判一边。与其假装能一刀切, 不如把不确定显式表达出来。
    DOMAIN_SCORE_CAUTION: float = 0.28

    # ===== Embedding Model =====
    EMBEDDING_MODEL_NAME: str = "embedding-3"
    # 向量维度。**0 表示自动探测**(从嵌入模型实际输出测量)。
    # 为什么不再写死: 原代码硬编码 2048, 而智谱 embedding-3 实际返回
    # **1024** 维 —— 这个不一致只在真正插入 Milvus 时才暴露
    # (MilvusException: the dim of field data is not equal to schema dim),
    # 而那时已经白跑了 107 次嵌入调用。自动探测 + 建库前校验可以避免这种浪费。
    # 只有在需要"锁死"维度(例如多环境对齐)时才显式赋值。
    EMBEDDING_DIM: int = 0

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
