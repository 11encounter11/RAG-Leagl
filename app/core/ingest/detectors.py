"""
结构探测器 —— 用确定性正则判断文本里有没有结构

为什么不用 LLM 判断:
    "这段文字里有没有『第X条』" 是可以用正则 100% 确定的事。
    让概率模型去猜, 只会引入不确定性。LLM 分类只作为最后兜底。

参见 docs/02a-文档解析与切分策略设计.md §三
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---- 中文数字 ----
_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_UNITS = {"十": 10, "百": 100, "千": 1000}

# 匹配中文/阿拉伯数字的字符集
_CN_NUM = "0-9零〇一二三四五六七八九十百两"

_WHITESPACE = " \t\u3000"


def cn_to_int(value: str) -> int | None:
    """中文数字/阿拉伯数字 -> int。失败返回 None。

    >>> cn_to_int("四十七")
    47
    >>> cn_to_int("一千二百六十")
    1260
    >>> cn_to_int("十")
    10
    >>> cn_to_int("47")
    47
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)

    total = 0
    section = 0
    number = 0
    for char in value:
        if char in _DIGITS:
            number = _DIGITS[char]
        elif char in _UNITS:
            unit = _UNITS[char]
            if number == 0:
                number = 1  # "十" 这种省略了前导一的写法
            section += number * unit
            number = 0
        else:
            return None
    return section + number


# ---- 结构正则 ----

# 条文标题: 行首的 "第X条" / "第X条之Y"
# 刻意保持简单: 不加花哨的负向断言, 因为正文里几乎必然含逗号,
# 任何"后面不能有逗号"的断言都会把真正的法条误杀。
RE_ARTICLE = re.compile(
    r"^[" + _WHITESPACE + r"]*(第\s*[" + _CN_NUM + r"]+\s*条(?:\s*之\s*[" + _CN_NUM + r"]+)?)",
    re.MULTILINE,
)

# 章/节/编标题: "第一章 总则"
RE_HEADING = re.compile(
    r"^[" + _WHITESPACE + r"]*(第\s*[" + _CN_NUM + r"]+\s*(?:编|章|节))["
    + _WHITESPACE + r"]*(.*)$",
    re.MULTILINE,
)

# 数字条款号: "1.1" / "1.1.1" / "1、"
RE_CLAUSE_NUM = re.compile(r"^[" + _WHITESPACE + r"]*(\d+(?:\.\d+)*)[" + _WHITESPACE + r"]*[、.．)）]", re.MULTILINE)

# 中文条目号: "一、" / "（一）"
RE_CLAUSE_CN = re.compile(r"^[" + _WHITESPACE + r"]*[（(]?([一二三四五六七八九十]{1,3})[)）]?[" + _WHITESPACE + r"]*[、.．]", re.MULTILINE)

# 问答
RE_QA_CN = re.compile(r"^[" + _WHITESPACE + r"]*(问|问题)\s*[:：]", re.MULTILINE)
RE_QA_EN = re.compile(r"^[" + _WHITESPACE + r"]*Q\s*[:：]", re.MULTILINE)

# 法律名称候选(行首独立成行的名字)
RE_LAW_TITLE_LINE = re.compile(
    r"^[ \t\u3000]*《?(中华人民共和国[\u4e00-\u9fff]{2,30}|[\u4e00-\u9fff]{2,20}(?:法|条例|办法|规定|解释|细则))》?[ \t\u3000]*$",
    re.MULTILINE,
)


@dataclass
class StructureSignals:
    """结构探测结果。"""

    article_count: int = 0
    heading_count: int = 0
    clause_num_count: int = 0
    clause_cn_count: int = 0
    qa_cn_count: int = 0
    qa_en_count: int = 0
    text_length: int = 0
    density: float = 0.0
    law_name: str | None = None
    headings: list[str] = field(default_factory=list)

    @property
    def has_article_structure(self) -> bool:
        return self.article_count >= 3

    @property
    def has_clause_structure(self) -> bool:
        return self.clause_num_count >= 3 or self.clause_cn_count >= 3

    @property
    def has_qa_structure(self) -> bool:
        return self.qa_cn_count >= 3 or self.qa_en_count >= 3

    def summary(self) -> dict:
        return {
            "article_count": self.article_count,
            "heading_count": self.heading_count,
            "clause_num_count": self.clause_num_count,
            "clause_cn_count": self.clause_cn_count,
            "qa_cn_count": self.qa_cn_count,
            "text_length": self.text_length,
            "law_name": self.law_name,
        }


def detect_structure(text: str, filename: str = "") -> StructureSignals:
    """扫描文本, 返回结构信号。"""
    text = text or ""
    signals = StructureSignals(text_length=len(text))

    signals.article_count = len(RE_ARTICLE.findall(text))
    signals.heading_count = len(RE_HEADING.findall(text))
    signals.clause_num_count = len(RE_CLAUSE_NUM.findall(text))
    signals.clause_cn_count = len(RE_CLAUSE_CN.findall(text))
    signals.qa_cn_count = len(RE_QA_CN.findall(text))
    signals.qa_en_count = len(RE_QA_EN.findall(text))

    signals.headings = []
    for match in RE_HEADING.finditer(text):
        number = re.sub(r"\s+", "", match.group(1))
        title = re.sub(r"[ \t\u3000]+", " ", match.group(2)).strip()
        signals.headings.append((number + " " + title).strip())

    # 结构密度: 每千字的结构标记数
    markers = signals.article_count + signals.heading_count
    signals.density = round(markers / max(len(text) / 1000.0, 0.001), 2)

    signals.law_name = extract_law_name(text, filename)
    return signals


def extract_law_name(text: str, filename: str = "") -> str | None:
    """提取法律/文档名称。

    优先取正文首部的标题行(更权威), 取不到再退回文件名。
    """
    head = "\n".join((text or "").splitlines()[:10])
    match = RE_LAW_TITLE_LINE.search(head)
    if match:
        return match.group(1).strip()

    if filename:
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename).strip()
        stem = re.sub(r"[（(].*?[)）]", "", stem).strip()
        if stem:
            return stem
    return None


def iter_heading_positions(text: str) -> list[tuple[int, str, str]]:
    """返回 [(位置, 层级类型(编/章/节), 完整标题)]。"""
    out: list[tuple[int, str, str]] = []
    for match in RE_HEADING.finditer(text):
        raw_number = re.sub(r"\s+", "", match.group(1))
        level = raw_number[-1]  # 编 / 章 / 节
        raw_title = re.sub(r"[ \t\u3000]+", " ", match.group(2)).strip()
        full_title = (raw_number + " " + raw_title).strip()
        out.append((match.start(), level, full_title))
    return out
