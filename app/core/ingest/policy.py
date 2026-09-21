"""
企业规章制度摄取器 —— 按编号层级切分

企业文档与法律法规的结构差异:
    法律法规: 编 -> 章 -> 节 -> 条   (稳定的国家级结构)
    企业制度: 章/条 混用 或 一、二、三 条目式 或 1.1 / 1.1.1 条款式
所以这份摄取器优先复用"第X条", 没有则退回中文条目号, 再没有则交给降级链。
"""
from __future__ import annotations

import re
from typing import Any

from app.core.ingest.base import (
    CONFIDENCE_DEGRADED,
    CONFIDENCE_FALLBACK,
    CONFIDENCE_HIGH,
    BaseIngestor,
    IngestResult,
    ParsedSection,
    normalize_text,
)
from app.core.ingest.detectors import (
    RE_ARTICLE,
    RE_CLAUSE_CN,
    RE_CLAUSE_NUM,
    cn_to_int,
    detect_structure,
    iter_heading_positions,
)
from app.core.ingest.metadata import build_section_extra

DEFAULT_MAX_CHARS = 1500
_SENTENCE_END = re.compile(r"(?<=[。；])")


class PolicyIngestor(BaseIngestor):
    """企业规章制度(员工手册 / 管理办法 / 操作规范)。"""

    doc_type = "policy"
    strategy = "clause_based"
    display_name = "企业规章制度（按编号层级切分）"

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self.max_chars = max_chars

    def detect(self, text: str, filename: str = "") -> float:
        text = normalize_text(text)
        signals = detect_structure(text, filename)
        # 有法条样式但不是法规(条数少) -> 更可能是企业制度
        if 1 <= signals.article_count < 3:
            return 0.7
        if signals.has_clause_structure and not signals.has_article_structure:
            return 0.6
        name = filename or ""
        if re.search(r"(手册|管理办法|管理制度|操作规范|流程|细则|规定)", name):
            return 0.65
        return 0.0

    def parse(self, text: str, meta: dict[str, Any] | None = None) -> IngestResult:
        meta = meta or {}
        text = normalize_text(text)
        doc_name = meta.get("doc_name") or meta.get("filename") or None

        sections = self._parse_by_article(text, meta, doc_name)
        if sections:
            return self.build_result(
                sections,
                doc_type=self.doc_type,
                strategy="clause_article_based",
                confidence=CONFIDENCE_HIGH,
                structure_detected=True,
                detail={"doc_name": doc_name, "mode": "article", "sections": len(sections)},
            )

        sections = self._parse_by_numbered_clause(text, meta, doc_name)
        if sections:
            return self.build_result(
                sections,
                doc_type=self.doc_type,
                strategy=self.strategy,
                confidence=CONFIDENCE_FALLBACK,
                structure_detected=True,
                detail={"doc_name": doc_name, "mode": "numbered", "sections": len(sections)},
            )

        return self.build_result(
            [],
            doc_type=self.doc_type,
            strategy=self.strategy,
            confidence=CONFIDENCE_DEGRADED,
            structure_detected=False,
            detail={"doc_name": doc_name, "reason": "no_clause_marker"},
        )

    # ---------- 模式一: 第X条 ----------
    def _parse_by_article(
        self, text: str, meta: dict[str, Any], doc_name: str | None
    ) -> list[ParsedSection]:
        matches = list(RE_ARTICLE.finditer(text))
        if not matches:
            return []

        headings = iter_heading_positions(text)
        cut_positions = sorted({h[0] for h in headings} | {m.start() for m in matches})

        sections: list[ParsedSection] = []
        level_stack: dict[str, str] = {}
        cursor = 0

        for match in matches:
            start = match.start()
            while cursor < len(headings) and headings[cursor][0] < start:
                _, level, title = headings[cursor]
                level_stack[level] = title
                if level == "编":
                    level_stack.pop("章", None)
                    level_stack.pop("节", None)
                elif level == "章":
                    level_stack.pop("节", None)
                cursor += 1

            end = self._next_cut(cut_positions, match.end())
            body = text[start:end].strip()
            if not body:
                continue

            section_path = " > ".join(
                level_stack[key] for key in ("编", "章", "节") if key in level_stack
            ) or None
            article_no = re.sub(r"\s+", "", match.group(1))

            for part_index, part in enumerate(self._split_long(body)):
                sections.append(
                    ParsedSection(
                        text=part,
                        article_no=article_no,
                        article_no_norm=self._normalize_article_no(article_no),
                        section_path=section_path,
                        seq=len(sections),
                        extra=build_section_extra(
                            meta, law_name=doc_name,
                            part_index=part_index, part_total=len(self._split_long(body)),
                        ),
                    )
                )
        return sections

    # ---------- 模式二: 一、 / 1.1 ----------
    def _parse_by_numbered_clause(
        self, text: str, meta: dict[str, Any], doc_name: str | None
    ) -> list[ParsedSection]:
        markers: list[tuple[int, str]] = []
        for match in RE_CLAUSE_NUM.finditer(text):
            markers.append((match.start(), match.group(1)))
        if len(markers) < 3:
            markers = []
            for match in RE_CLAUSE_CN.finditer(text):
                markers.append((match.start(), match.group(1) + "、"))
        if len(markers) < 3:
            return []

        markers.sort(key=lambda item: item[0])
        sections: list[ParsedSection] = []
        for index, (start, clause_no) in enumerate(markers):
            end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            for part_index, part in enumerate(self._split_long(body)):
                sections.append(
                    ParsedSection(
                        text=part,
                        clause_no=clause_no,
                        section_path=None,
                        seq=len(sections),
                        extra=build_section_extra(
                            meta, law_name=doc_name,
                            part_index=part_index, part_total=len(self._split_long(body)),
                        ),
                    )
                )
        return sections

    # ---------- 工具 ----------
    @staticmethod
    def _next_cut(cut_positions: list[int], after: int) -> int:
        for position in cut_positions:
            if position > after:
                return position
        return 1 << 60

    @staticmethod
    def _normalize_article_no(article_no: str) -> int | None:
        match = re.search(r"第\s*([0-9零〇一二三四五六七八九十百两]+)\s*条", article_no)
        return cn_to_int(match.group(1)) if match else None

    def _split_long(self, body: str) -> list[str]:
        if len(body) <= self.max_chars:
            return [body]
        chunks: list[str] = []
        buffer = ""
        for sentence in _SENTENCE_END.split(body):
            if not sentence:
                continue
            if len(buffer) + len(sentence) <= self.max_chars:
                buffer += sentence
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = sentence
        if buffer:
            chunks.append(buffer)
        return chunks or [body]
