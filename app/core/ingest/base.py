"""
摄取器基类与数据结构

职责边界:
    Loader  : 文件格式(docx/pdf/xlsx/...) -> 纯文本
    Ingestor: 纯文本 -> 带结构信息的片段(ParsedSection)

这样切分逻辑与文件格式解耦, 一种文档类型只需实现一个 Ingestor。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# 置信度等级: 高 / 降级 / 完全退化
CONFIDENCE_HIGH = "high"
CONFIDENCE_FALLBACK = "fallback"
CONFIDENCE_DEGRADED = "degraded"


def normalize_text(text: str) -> str:
    """统一换行符。

    真实文件里 CRLF 很常见, 不归一化会让 "第X章 标题" 这类行尾捕获
    带上 \\r, 进而污染章节路径与条约号。
    """
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")



@dataclass
class ParsedSection:
    """一个切分单元。法律场景下通常就是"一条法条"。

    字段与 docs/01-需求文档.md §F3.2 的元数据清单对齐。
    """

    text: str

    # ---- 结构坐标 ----
    article_no: str | None = None       # "第四十七条"
    article_no_norm: int | None = None  # 47
    chapter_path: str | None = None     # "第四章 劳动合同的解除和终止"
    section_path: str | None = None     # 通用层级路径(企业制度用)
    clause_no: str | None = None        # "1.1" / "一、" (合同/制度用)
    question: str | None = None         # FAQ 用: 独立存问题, 便于 query 匹配

    # ---- 顺序 ----
    seq: int = 0

    # ---- 其他 ----
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """转成可以写进向量库元数据的扁平字典(去掉 None)。"""
        meta: dict[str, Any] = {"seq": self.seq}
        for key, value in (
            ("article_no", self.article_no),
            ("article_no_norm", self.article_no_norm),
            ("chapter_path", self.chapter_path),
            ("section_path", self.section_path),
            ("clause_no", self.clause_no),
            ("question", self.question),
        ):
            if value is not None:
                meta[key] = value
        meta.update(self.extra)
        return meta


@dataclass
class IngestResult:
    """一次摄取的结果。strategy/confidence 必须落库, 用于后期质量归因。"""

    sections: list[ParsedSection]
    doc_type: str
    strategy: str
    confidence: str
    structure_detected: bool
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        return len(self.sections)

    def is_degraded(self) -> bool:
        return self.confidence != CONFIDENCE_HIGH


class BaseIngestor(ABC):
    """摄取器基类。

    子类必须定义 doc_type(注册键) 与 strategy(留痕用), 并实现 parse()。
    detect() 给出"这份文本像不像我这类文档"的置信度(0~1)。
    """

    doc_type: str = ""
    strategy: str = ""
    display_name: str = ""

    # ---- 类型判定 ----
    def detect(self, text: str, filename: str = "") -> float:
        """返回 0~1 的置信度。默认实现返回 0(不主动认领)。"""
        return 0.0

    # ---- 解析 ----
    @abstractmethod
    def parse(self, text: str, meta: dict[str, Any] | None = None) -> IngestResult:
        """把纯文本切成带结构信息的片段。"""

    # ---- 工具 ----
    @staticmethod
    def finalize_sequence(sections: list[ParsedSection]) -> list[ParsedSection]:
        """统一编号 seq, 并丢弃空白片段。"""
        cleaned = [s for s in sections if s.text and s.text.strip()]
        for index, section in enumerate(cleaned):
            section.seq = index
        return cleaned

    @staticmethod
    def build_result(
        sections: list[ParsedSection],
        *,
        doc_type: str,
        strategy: str,
        confidence: str,
        structure_detected: bool,
        detail: dict[str, Any] | None = None,
    ) -> IngestResult:
        return IngestResult(
            sections=BaseIngestor.finalize_sequence(sections),
            doc_type=doc_type,
            strategy=strategy,
            confidence=confidence,
            structure_detected=structure_detected,
            detail=detail or {},
        )
