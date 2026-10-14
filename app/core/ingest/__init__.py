"""
文档摄取模块 —— 多类型解析与切分策略路由

设计依据: docs/02a-文档解析与切分策略设计.md

核心思想:
    切分策略绑定「文档类型」, 而不是全局一套。
    类型判定优先用确定性规则(结构探测/文件名), 最后才用 LLM。
    每一级都要有降级, 且降级必须留痕。
"""
from app.core.ingest.base import (
    CONFIDENCE_DEGRADED,
    CONFIDENCE_FALLBACK,
    CONFIDENCE_HIGH,
    BaseIngestor,
    IngestResult,
    ParsedSection,
    normalize_text,
)
from app.core.ingest.metadata import (
    DOC_TYPE_FAQ,
    DOC_TYPE_GENERIC,
    DOC_TYPE_STATUTE,
    REQUIRED_CHUNK_FIELDS,
    missing_required_fields,
)
from app.core.ingest.nodes import build_contextual_prefix, build_text_nodes
from app.core.ingest.registry import (
    REGISTRY,
    get_ingestor,
    ingest_text,
    register,
    resolve_ingestor,
)

__all__ = [
    "BaseIngestor",
    "IngestResult",
    "ParsedSection",
    "normalize_text",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_FALLBACK",
    "CONFIDENCE_DEGRADED",
    "DOC_TYPE_STATUTE",
    "DOC_TYPE_FAQ",
    "DOC_TYPE_GENERIC",
    "REQUIRED_CHUNK_FIELDS",
    "missing_required_fields",
    "build_text_nodes",
    "build_contextual_prefix",
    "REGISTRY",
    "register",
    "resolve_ingestor",
    "get_ingestor",
    "ingest_text",
]
