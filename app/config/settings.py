"""
应用配置管理
从 .env 文件读取配置，提供全局配置对象
"""
import os

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
    # 必须与 docker-compose.yml 的建库名一致(现状 P1-18: 默认值曾为 rag_system,
    # 与 compose 的 legal_rag 不匹配, 未设置 .env 时启动即连不上)。
    MYSQL_DATABASE: str = "legal_rag"

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
    # 重排之后的**准入阈值**：只判本批候选的第 1 名，达标就保留、最终条数交给 top_k。
    # 比的是 Cross-Encoder 归一化后的相关度分（0~1），**不是余弦相似度** ——
    # 名字是历史包袱（.env/API/前端都叫它 SIMILARITY_THRESHOLD），语义见
    # retriever.filter_by_similarity_threshold 的说明。
    # 数值 0.1 来自 2026-09-22 的四组对照（见 07-P0端到端验收与重排收益.md §3），
    # 但当时语料以《劳动法》107 条为主、切分也不同；现在 7 部法 / 703 条，绝对分的分布已不可比。
    # 所以它只能当"这一批值不值得答"的粗闸门，不能当每条片段的及格线（P2-20）。
    SIMILARITY_THRESHOLD: float = 0.1
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
    # 是否让 HuggingFace 走离线模式(只读本地缓存, 不发网络请求)。
    #
    # 为什么默认开: sentence-transformers 加载本地已缓存的模型时, 仍会先向
    # huggingface.co 发 HEAD 请求核对文件是否有更新。国内网络下这个请求会超时,
    # 然后按 1s/2s/4s... 退避重试 5 次 —— 表现为"启动卡住好几分钟",
    # 很容易被误判成 OOM 或死机(见 docs/06-P0测试执行报告.md §3.1)。
    #
    # 代价: 模型**未**缓存时会直接报错而不是自动下载。需要首次下载时,
    # 在 .env 里临时设为 false 即可。
    HF_HUB_OFFLINE: bool = True

    # ===== Agentic RAG Settings =====
    REFLECTION_THRESHOLD: float = 0.6
    MAX_RETRIES: int = 2
    # 各节点的模型(2026-10-14 起统一用 DeepSeek; 智谱的额度专供 embedding)
    QUERY_REWRITE_MODEL: str = "deepseek-v4.1-flash"
    HYDE_MODEL: str = "deepseek-v4.1-flash"
    # 反思节点专用的模型。刻意与改写模型分开配置 —— 用同一个模型评估自己
    # 的检索结果属于自评自夸, 至少要让这件事是显式可调的。
    REFLECTION_MODEL: str = "deepseek-v4.1-flash"

    # ===== 功能开关 =====
    # P0 阶段先放在配置里(服务端生效、可验证); P3 迁移到数据库后台管理。
    # 关掉某个开关后, 工作流会真的不执行那个节点 —— 不是"前端不传参"那种伪生效。
    ENABLE_AGENTIC_RAG: bool = True
    ENABLE_QUERY_REWRITE: bool = True
    ENABLE_HYDE: bool = False          # 默认关: 成本高, 收益尚未被评测验证
    ENABLE_RERANK: bool = True
    ENABLE_CITATION_VERIFY: bool = True
    # 时效过滤开关（T-27 / T-28，需求 F4.2）。
    #
    # 开（默认）：向量召回时把 `validity_status == "effective"` 下推到 Milvus 查询表达式，
    #             已修订/已废止的版本**根本不会进入候选** —— 不依赖 LLM 自觉忽略旧条文。
    # 关：     不做时效过滤，旧版本也会被召回（用于对照验证"过滤确实生效"，
    #             以及 F4.3「用户明确查历史法条」的场景）。
    #
    # 为什么放在检索层而不是生成层：靠提示词让模型"别看废止条文"是概率保证，
    # 而在查询表达式里过滤是**确定性保证**（见 02-设计文档.md §6.1 的说明）。
    ENABLE_VALIDITY_FILTER: bool = True

    # ===== 拒答闸门(独立判据) =====
    # 为什么必须独立于重排器: 拒答是法律场景的安全底线, 而重排器是可选的、
    # 可能在环境里加载不起来的组件。把底线建立在可选组件之上, 结果是
    # "重排器加载失败 -> 该拒答的全都不拒答"(实测拒答准确率 0%)。
    #
    # 判据: 向量检索的**最高原始相似度**。它与知识库主题的相关程度直接对应,
    # 不经过 RRF 融合、不依赖重排器。
    #
    # 阈值是标定出来的, 不是拍的: 用 scripts/calibrate_domain_gate.py 在评测集上
    # 跑出分布, 再选点。
    #
    # 2026-10-14 首标(语料从 1 部法扩到 7 部/703 片段后, 原 0.24/0.28 失效 ——
    # 硬拒答只剩 2/16、且 6 条超纲题完全放行):
    #     库内 n=36  min=0.2779  中位=0.3468  max=0.4201
    #     超纲 n=16  min=0.2237  中位=0.2698  max=0.2931
    #   两分布重叠于 [0.2779, 0.2931], 单阈值切不开 -> 维持两档设计。
    #   当时选点: REJECT=0.26 / CAUTION=0.30(硬拒答 7/16、有警示 16/16、误杀 0/36)。
    #
    # 2026-10-15 重标 —— **嵌入内容修复后必须重标**:
    #   元数据不再混进待嵌入文本(见 ingest/nodes.py 与 04-P1 的修复记录),
    #   相似度分布整体上移, 而且**两分布不再重叠**:
    #     库内 n=36  min=0.3504  中位=0.4078  max=0.4943
    #     超纲 n=16  min=0.2706  中位=0.3094  max=0.3399
    #   空档 [0.3399, 0.3504], 宽 0.0105。旧值 0.26/0.30 落在这个空档**之下**,
    #   于是硬拒答掉到 0/16 —— 超纲题会被自信地用劳动法回答。这是修复带来的
    #   副作用, 也是"绝对阈值必须随分数分布重标"的活例子。
    #   新选点:
    #     REJECT  = 0.345 = 空档中点: 超纲 16/16 全拦住、库内 0/36 误杀,
    #                        两侧各留 0.005 余量, 比贴边取值更抗抖动
    #     CAUTION = 0.36  = 库内最低值上方, 让最贴近边界的那道库内问题
    #                       得到核对提示(L21 0.3504 / L24 0.3596 两道)
    #   效果(实测): 硬拒答 16/16, 有警示 16/16, 误杀库内 0/36, 库内低置信 2/36。
    #
    # **余量只有 0.0105**: 语料或评测集一变(尤其新增超纲题)必须重跑
    # scripts/calibrate_domain_gate.py。更稳的做法是改用**相对判据**
    # (如 top1 与 top2 的差值, 或对语料分布做标准化)而不是绝对余弦值 ——
    # bge 这类模型在 [0.3, 0.5] 这个窄带里普遍偏高, 绝对阈值天然脆弱。
    ENABLE_DOMAIN_GATE: bool = True
    # 低于此值: 判定为"知识库外", 直接拒答
    DOMAIN_SCORE_REJECT: float = 0.345
    # 低于此值(但不低于 REJECT): 照常回答, 但标记为低置信, 提示用户核对
    # 采用两档而不是单阈值, 是因为实测下两个分布**区间重叠**:
    # 任何单阈值都必然误判一边。与其假装能一刀切, 不如把不确定显式表达出来。
    DOMAIN_SCORE_CAUTION: float = 0.36

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
    # CORS 允许的来源(逗号分隔)。
    # 刻意不用 "*": 与 allow_credentials=True 组合会被浏览器直接拒绝,
    # 且等价于完全放开(见 02-设计文档.md §13 P1 #20)。
    # 默认只放本机前端(Streamlit 默认端口 8501)。
    CORS_ALLOW_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

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

    @property
    def cors_origins(self) -> list[str]:
        """把逗号分隔的 CORS 来源解析成列表。"""
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()

# 必须在任何 huggingface_hub 相关库被导入**之前**设置:
# huggingface_hub 在 import 时就把 HF_HUB_OFFLINE 读进常量, 之后再设无效。
# 本模块是所有入口(应用、脚本、测试)最先导入的模块之一, 放在这里最稳妥 ——
# 靠"记得在命令行手动加环境变量"是守不住的, 那条纪律早晚会被忘掉。
if settings.HF_HUB_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
