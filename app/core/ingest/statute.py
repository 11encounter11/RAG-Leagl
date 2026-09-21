"""
法律法规摄取器 —— 按"条"切分

这是本项目的核心切分策略。为什么不用定长切分:
    1. 定长会把一条法条拦腰切断, 检索命中半条时 LLM 看到的是没有主语的句子
    2. 法条之间有依赖("前款规定的..."、"依照本法第四十六条"), 按条切才留得住
    3. 条文号是引用溯源的坐标, 定长切分拿不到

产出: 一条法条 = 一个 ParsedSection, 带 article_no / article_no_norm / chapter_path
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
from app.core.ingest.detectors import (
    RE_ARTICLE,
    cn_to_int,
    detect_structure,
    iter_heading_positions,
)
from app.core.ingest.metadata import build_section_extra

# 单条超过这个长度就继续按句号切开。
# 为什么必须切: embedding 模型有最大输入长度, 超长内容会被"静默截断",
# 你以为嵌了整条法条, 实际只嵌了前半段。
DEFAULT_MAX_CHARS = 1500

_SENTENCE_END = re.compile(r"(?<=[。；])")


def strip_article_prefix(raw: str) -> str:
    """把 "第 一 条" 归一化成 "第一条"。"""
    return re.sub(r"\s+", "", raw or "")


class StatuteIngestor(BaseIngestor):
    """法律法规 / 司法解释 / 地方性法规 的按条切分。"""

    doc_type = "statute"
    strategy = "article_based"
    display_name = "法律法规（按条切分）"

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self.max_chars = max_chars

    # ---------- 类型判定 ----------
    def detect(self, text: str, filename: str = "") -> float:
        signals = detect_structure(normalize_text(text), filename)
        if signals.article_count < 3:
            return 0.0
        score = min(0.99, 0.6 + signals.article_count / 200.0)
        # 企业内部的制度文件也常用"第X条", 但文件名会暴露它。
        # 这里主动压低得分, 让 PolicyIngestor 有机会胜出,
        # 从而避免把员工手册当成国家法规处理(那会导致 law_name/章节路径错位)。
        if re.search(r"(手册|管理制度|管理办法|操作规范|操作流程|员工|考勤|报销|薪酬|绩效)", filename or ""):
            score *= 0.5
        return score


    # ---------- 解析 ----------
    def parse(self, text: str, meta: dict[str, Any] | None = None) -> IngestResult:
        meta = meta or {}
        text = normalize_text(text)
        signals = detect_structure(text, meta.get("filename", ""))
        law_name = meta.get("law_name") or signals.law_name

        articles = list(RE_ARTICLE.finditer(text))
        if not articles:
            return self.build_result(
                [],
                doc_type=self.doc_type,
                strategy=self.strategy,
                confidence=CONFIDENCE_DEGRADED,
                structure_detected=False,
                detail={"reason": "no_article_marker"},
            )

        headings = iter_heading_positions(text)
        # 所有"切割点": 章节标题位置 + 条文起点。
        # 用途是界定每条法条的正文范围 —— 不能简单地"到下一个条文为止",
        # 否则夹在两条之间的 "第二章 xxx" 标题会被算进上一条的正文里。
        cut_positions = sorted({h[0] for h in headings} | {a.start() for a in articles})

        sections: list[ParsedSection] = []
        level_stack: dict[str, str] = {}
        heading_cursor = 0

        for match in articles:
            start = match.start()

            # 推进章节指针, 记录当前所处的编/章/节
            while heading_cursor < len(headings) and headings[heading_cursor][0] < start:
                _, level, title = headings[heading_cursor]
                level_stack[level] = title
                if level == "编":
                    level_stack.pop("章", None)
                    level_stack.pop("节", None)
                elif level == "章":
                    level_stack.pop("节", None)
                heading_cursor += 1

            end = self._next_cut(cut_positions, match.end())
            body = text[start:end].strip()
            if not body:
                continue

            chapter_path = " > ".join(
                level_stack[key] for key in ("编", "章", "节") if key in level_stack
            ) or None

            article_no = strip_article_prefix(match.group(1))
            article_no_norm = self._normalize_article_no(article_no)
            parts = self._split_long(body)

            for part_index, part in enumerate(parts):
                sections.append(
                    ParsedSection(
                        text=part,
                        article_no=article_no,
                        article_no_norm=article_no_norm,
                        chapter_path=chapter_path,
                        seq=len(sections),
                        extra=build_section_extra(
                            meta,
                            law_name=law_name,
                            part_index=part_index,
                            part_total=len(parts),
                        ),
                    )
                )

        confidence = CONFIDENCE_HIGH if sections else CONFIDENCE_DEGRADED
        return self.build_result(
            sections,
            doc_type=self.doc_type,
            strategy=self.strategy,
            confidence=confidence,
            structure_detected=True,
            detail={
                "law_name": law_name,
                "article_count": len(articles),
                "chapter_count": len(headings),
                "sections": len(sections),
                "long_articles_split": sum(
                    1 for s in sections if s.extra.get("part_total", 1) > 1
                ),
            },
        )

    # ---------- 内部工具 ----------
    @staticmethod
    def _next_cut(cut_positions: list[int], after: int) -> int:
        """返回 after 之后的第一个切割点(无则返回极大值)。"""
        for position in cut_positions:
            if position > after:
                return position
        return 1 << 60

    @staticmethod
    def _normalize_article_no(article_no: str) -> int | None:
        """"第四十七条" -> 47 ; "第十条之二" -> 10"""
        match = re.search(r"第\s*([0-9零〇一二三四五六七八九十百两]+)\s*条", article_no)
        if not match:
            return None
        return cn_to_int(match.group(1))

    def _split_long(self, body: str) -> list[str]:
        """超长法条按句子继续切, 避免被 embedding 模型静默截断。"""
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
