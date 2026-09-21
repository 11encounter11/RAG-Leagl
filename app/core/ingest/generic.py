"""
通用降级摄取器 —— 结构感知失败时的兜底

降级链的最后一环(见 docs/02a-文档解析与切分策略设计.md §四):
    ① 结构感知(按条 / 按条款)
        ↓ 探测失败
    ② 本模块: 递归分隔符切分(段落 -> 换行 -> 句号 -> 逗号 -> 定长 token 窗口)
        ↓
    永不返回空结果

为什么必须存在: 企业上传的文档一定会有格式错乱的。宁可切得差,
也不能让整个上传失败 —— 但要如实标记 confidence 为 fallback/degraded,
这样后期做质量归因时能一键筛出"所有走了降级的文档"。
"""
from __future__ import annotations

from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from app.config.settings import settings
from app.core.ingest.base import (
    CONFIDENCE_DEGRADED,
    CONFIDENCE_FALLBACK,
    BaseIngestor,
    IngestResult,
    ParsedSection,
    normalize_text,
)
from app.core.ingest.metadata import build_section_extra

# SentenceSplitter 的默认分隔符就是"递归"顺序:
#   \n\n -> \n -> 空格 -> 空(退化为 token 窗口)
# 所以它天然实现了降级链的 ② 与 ③ 两级。


class GenericIngestor(BaseIngestor):
    """无结构文本的兜底切分。"""

    doc_type = "generic"
    strategy = "recursive_split"
    display_name = "通用文本（递归分隔符 + 定长兜底）"

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def detect(self, text: str, filename: str = "") -> float:
        # 它从不"主动认领", 只做最后一环
        return 0.0

    def parse(self, text: str, meta: dict[str, Any] | None = None) -> IngestResult:
        meta = meta or {}
        text = normalize_text(text)

        if not text.strip():
            return self.build_result(
                [],
                doc_type=self.doc_type,
                strategy=self.strategy,
                confidence=CONFIDENCE_DEGRADED,
                structure_detected=False,
                detail={"reason": "empty_text"},
            )

        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents([Document(text=text)])

        sections: list[ParsedSection] = []
        for node in nodes:
            content = (node.text or "").strip()
            if not content:
                continue
            sections.append(
                ParsedSection(
                    text=content,
                    seq=len(sections),
                    extra=build_section_extra(meta, law_name=meta.get("doc_name")),
                )
            )

        confidence = CONFIDENCE_FALLBACK if sections else CONFIDENCE_DEGRADED
        return self.build_result(
            sections,
            doc_type=self.doc_type,
            strategy=self.strategy,
            confidence=confidence,
            structure_detected=False,
            detail={
                "reason": "structure_not_detected",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        )
