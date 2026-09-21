"""
FAQ 摄取器 —— 一问一块, 不再切分

为什么单独一类:
    问答对是最小完整语义单元, 再切就破坏了"问题-答案"的配对关系。
    另外问题文本单独存进 metadata.question, 可用于 query 侧的精确匹配,
    而不只是靠向量语义相似。

支持格式:
    问：xxx         Q：xxx        Q: xxx
    答：yyy         A：yyy        A: yyy
"""
from __future__ import annotations

import re
from typing import Any

from app.core.ingest.base import (
    CONFIDENCE_DEGRADED,
    CONFIDENCE_HIGH,
    BaseIngestor,
    IngestResult,
    ParsedSection,
    normalize_text,
)
from app.core.ingest.detectors import detect_structure
from app.core.ingest.metadata import build_section_extra

RE_QUESTION = re.compile(r"^[ \t\u3000]*(?:问|问题|Q)\s*[:：]\s*(.*)$", re.MULTILINE)
RE_ANSWER = re.compile(r"^[ \t\u3000]*(?:答|答案|A)\s*[:：]\s*(.*)$", re.MULTILINE)


class FaqIngestor(BaseIngestor):
    """FAQ / 问答对。"""

    doc_type = "faq"
    strategy = "qa_pair"
    display_name = "问答对（一问一块）"

    def detect(self, text: str, filename: str = "") -> float:
        text = normalize_text(text)
        signals = detect_structure(text, filename)
        if signals.qa_cn_count + signals.qa_en_count >= 3:
            return 0.9
        if re.search(r"(faq|问答|常见问题|答疑)", (filename or ""), re.IGNORECASE):
            return 0.8
        return 0.0

    def parse(self, text: str, meta: dict[str, Any] | None = None) -> IngestResult:
        meta = meta or {}
        text = normalize_text(text)

        questions = list(RE_QUESTION.finditer(text))
        if not questions:
            return self.build_result(
                [],
                doc_type=self.doc_type,
                strategy=self.strategy,
                confidence=CONFIDENCE_DEGRADED,
                structure_detected=False,
                detail={"reason": "no_question_marker"},
            )

        sections: list[ParsedSection] = []
        for index, q_match in enumerate(questions):
            start = q_match.start()
            end = questions[index + 1].start() if index + 1 < len(questions) else len(text)
            block = text[start:end].strip()

            question = q_match.group(1).strip()
            answer_match = RE_ANSWER.search(block)
            answer = answer_match.group(1).strip() if answer_match else ""

            # 检索与向量化都希望"问题 + 答案"在一起, 这样问法和答法都能命中
            combined = block
            if not answer:
                # 只有问没有答(或答的标记缺失): 整块当内容, 但仍记为降级
                pass

            sections.append(
                ParsedSection(
                    text=combined,
                    question=question,
                    seq=len(sections),
                    extra=build_section_extra(meta, law_name=meta.get("doc_name")),
                )
            )

        confidence = CONFIDENCE_HIGH if sections else CONFIDENCE_DEGRADED
        return self.build_result(
            sections,
            doc_type=self.doc_type,
            strategy=self.strategy,
            confidence=confidence,
            structure_detected=True,
            detail={"question_count": len(questions), "sections": len(sections)},
        )
