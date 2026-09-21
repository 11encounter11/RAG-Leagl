"""
回答生成相关模块: 提示词模板、引用格式、引用校验、拒答。
"""
from app.core.generate.citation import (
    CITATION_INSTRUCTION,
    VALIDITY_LABELS,
    build_context_block,
    build_sources,
    format_citation_label,
)
from app.core.generate.prompt import build_system_prompt

__all__ = [
    "CITATION_INSTRUCTION",
    "VALIDITY_LABELS",
    "build_context_block",
    "build_sources",
    "format_citation_label",
    "build_system_prompt",
]
