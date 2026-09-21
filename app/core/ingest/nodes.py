"""
ParsedSection -> TextNode 组装

这一层负责把摄取结果变成可以入库的节点, 并兑现元数据契约
(docs/01-需求文档.md §F3.2)。分层理由:

    Ingestor  只关心"结构在哪"(条文号/章节路径)
    本模块    只关心"元数据要齐、格式要能入库"
    VectorStore 只关心"怎么存怎么查"

contextual prefix 说明:
    向量化的内容里会拼上 "《法名》> 章节 > 条文号" 前缀。
    原因: 用户问"解除劳动合同能拿多少钱", 而条文原文里可能根本不出现
    "解除"二字(它只出现在章节标题里)。前缀让检索能命中。
    前缀同时保留在 text 里, 因为 LLM 生成与 BM25 检索都受益于它;
    干净的原文另存 display_text, 供引用溯源弹窗展示。
    参见 docs/02a-文档解析与切分策略设计.md §4.4
"""
from __future__ import annotations

from typing import Any

from llama_index.core.schema import TextNode

from app.core.ingest.base import IngestResult, ParsedSection
from app.core.ingest.metadata import (
    REQUIRED_CHUNK_FIELDS,
    build_document_metadata,
    missing_required_fields,
)

# 必填的字符串类字段: 值为 None 时用空串占位。
# 为什么不用 None: Milvus 标量字段对 NULL 支持不佳, 且过滤表达式
# 统一用 == "" 判断"不限定"比处理 NULL 简单得多。
_STRING_REQUIRED = {
    "tenant_id",
    "kb_id",
    "document_id",
    "doc_type",
    "source_file",
    "filename",
    "chunk_strategy",
    "strategy_confidence",
    "validity_status",
}


def build_contextual_prefix(section: ParsedSection, law_name: str | None) -> str:
    """拼出用于增强检索的结构前缀(不改变原文语义, 只补上下文)。"""
    parts: list[str] = []
    if law_name:
        parts.append(f"《{law_name}》")
    if section.chapter_path:
        parts.append(section.chapter_path)
    if section.section_path:
        parts.append(section.section_path)
    if section.article_no:
        parts.append(section.article_no)
    if section.clause_no:
        parts.append(f"第{section.clause_no}条" if section.clause_no.isdigit() else section.clause_no)
    if not parts:
        return ""
    return " > ".join(parts)


def _sanitize(metadata: dict[str, Any]) -> dict[str, Any]:
    """把元数据整理成 Milvus 能存的形式。"""
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            if key in _STRING_REQUIRED:
                cleaned[key] = ""
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    # 必填字段即使从未出现也要占位, 保证契约可被验证
    for field in REQUIRED_CHUNK_FIELDS:
        if field not in cleaned:
            cleaned[field] = "" if field in _STRING_REQUIRED else 0
    return cleaned


def build_text_nodes(
    result: IngestResult,
    *,
    doc_meta: dict[str, Any] | None = None,
    with_prefix: bool = True,
) -> list[TextNode]:
    """把摄取结果转成可入库的 TextNode 列表。"""
    doc_meta = doc_meta or {}
    doc_level = build_document_metadata(doc_meta)

    nodes: list[TextNode] = []
    for index, section in enumerate(result.sections):
        metadata: dict[str, Any] = {}
        metadata.update(doc_level)
        metadata.update(section.extra)          # extra 覆盖同名的文档级字段
        metadata.update(section.to_metadata())  # 结构坐标

        # 权威值: 摄取器实际判定出来的类型 / 策略 / 置信度
        metadata["doc_type"] = result.doc_type
        metadata["chunk_strategy"] = result.strategy
        metadata["strategy_confidence"] = result.confidence
        metadata["structure_detected"] = result.structure_detected
        metadata["chunk_index"] = index
        metadata["section_seq"] = section.seq

        law_name = metadata.get("law_name") or doc_meta.get("law_name")
        prefix = build_contextual_prefix(section, law_name) if with_prefix else ""
        text = f"{prefix}\n{section.text}" if prefix else section.text
        metadata["display_text"] = section.text

        node = TextNode(text=text, metadata=_sanitize(metadata))
        missing = missing_required_fields(node.metadata)
        if missing:
            raise ValueError(f"chunk 元数据缺字段 {missing}: {node.metadata}")
        nodes.append(node)

    return nodes
