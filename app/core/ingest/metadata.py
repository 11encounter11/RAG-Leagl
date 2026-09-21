"""
chunk 元数据契约 —— 唯一事实来源

对应 docs/01-需求文档.md §F3.2 的字段清单。
任何新增切分策略都必须通过本模块生成元数据, 不允许各写各的,
否则"字段齐全"这条验收标准无法被验证。

注意: 缺失元数据的后果是对应能力直接失效 ——
    缺 article_no  -> 无法引用溯源
    缺 document_id -> 无法删除文档(孤儿向量)
    缺 validity    -> 可能引用已废止法条
    缺 region_code -> C 端可能给出错误地域的答案
"""
from __future__ import annotations

from typing import Any

# 必须存在于每条 chunk 上的字段(值为 None 也允许, 但键必须出现)
REQUIRED_CHUNK_FIELDS: tuple[str, ...] = (
    "tenant_id",
    "kb_id",
    "document_id",
    "doc_type",
    "source_file",
    "filename",
    "chunk_index",
    "section_seq",
    "chunk_strategy",
    "strategy_confidence",
    "structure_detected",
    "validity_status",
)

# 结构类字段(并非所有文档类型都有, 例如 FAQ 没有条文号)
OPTIONAL_STRUCTURE_FIELDS: tuple[str, ...] = (
    "article_no",
    "article_no_norm",
    "chapter_path",
    "section_path",
    "clause_no",
    "question",
    "law_name",
    "version_id",
    "version_label",
    "effective_date",
    "expiry_date",
    "region_code",
    "part_index",
    "part_total",
)

VALIDITY_EFFECTIVE = "effective"   # 现行有效
VALIDITY_AMENDED = "amended"       # 已修订
VALIDITY_REPEALED = "repealed"     # 已废止
VALIDITY_PENDING = "pending"       # 尚未生效

# 文档类型
DOC_TYPE_STATUTE = "statute"       # 法律法规 / 司法解释
DOC_TYPE_POLICY = "policy"         # 企业规章制度
DOC_TYPE_FAQ = "faq"               # 问答对
DOC_TYPE_GENERIC = "generic"       # 兜底(无结构文本)


def build_document_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """从上传上下文里抽出"整篇文档级"的元数据, 供每条 chunk 复用。"""
    meta = meta or {}
    return {
        "tenant_id": meta.get("tenant_id", "platform"),
        "kb_id": meta.get("kb_id", "public"),
        "document_id": meta.get("document_id", ""),
        "doc_type": meta.get("doc_type", ""),
        "source_file": meta.get("source_file", ""),
        "filename": meta.get("filename", ""),
        "version_id": meta.get("version_id"),
        "version_label": meta.get("version_label"),
        # ---- 时效: P0 先落占位值, P1 接真实法规版本数据 ----
        "effective_date": meta.get("effective_date"),
        "expiry_date": meta.get("expiry_date"),
        "validity_status": meta.get("validity_status", VALIDITY_EFFECTIVE),
        # ---- 地域: 空串表示"全国性/不限定", 用空串而非 None 是为了
        #      Milvus 标量字段对 NULL 支持不佳, 过滤表达式的写法也更统一 ----
        "region_code": meta.get("region_code", ""),
    }


def build_section_extra(
    meta: dict[str, Any] | None,
    *,
    law_name: str | None = None,
    part_index: int | None = None,
    part_total: int | None = None,
) -> dict[str, Any]:
    """组装"本条特有"的元数据。

    刻意不把文档级元数据(租户/知识库/文档ID/时效/地域)放进来。
    理由: 那会让同一份文档的元数据出现两个来源 —— 摄取器在 parse() 时
    捕获一份, 调用方在组装节点时又想注入一份。两者不一致时, 摄取器里的
    旧值会静默覆盖调用方的新值(曾经因此把 tenant_id 写成默认值)。
    文档级元数据统一由 nodes.build_text_nodes 注入, 这里只放本条特有的。
    """
    extra: dict[str, Any] = {}
    if law_name:
        extra["law_name"] = law_name
    if part_index is not None:
        extra["part_index"] = part_index
    if part_total is not None:
        extra["part_total"] = part_total
    return extra


def missing_required_fields(metadata: dict[str, Any]) -> list[str]:
    """返回缺失的必填字段(用于测试与入库前自检)。"""
    return [field for field in REQUIRED_CHUNK_FIELDS if field not in metadata]
