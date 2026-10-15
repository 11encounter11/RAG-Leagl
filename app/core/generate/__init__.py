"""
回答生成相关模块: 提示词模板、引用格式、引用校验、拒答。
"""
from app.core.generate.citation import (
    CITATION_INSTRUCTION,
    VALIDITY_LABELS,
    authoritative_article_id,
    build_context_block,
    build_sources,
    extract_citation_indices,
    format_citation_label,
    verify_citation_existence,
    verify_citations,
)
from app.core.generate.prompt import build_system_prompt

__all__ = [
    "CITATION_INSTRUCTION",
    "VALIDITY_LABELS",
    "authoritative_article_id",
    "build_context_block",
    "build_sources",
    "extract_citation_indices",
    "format_citation_label",
    "verify_citation_existence",
    "verify_citations",
    "build_system_prompt",
]
