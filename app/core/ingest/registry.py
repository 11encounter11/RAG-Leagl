"""
摄取器注册表与路由

设计依据: docs/02a-文档解析与切分策略设计.md §三

判定优先级(从最可靠到最自动):
    ① user_hint     上传时用户指定          <- 主路径, 你自己知道传的是什么
    ② 结构探测器     正则判断有无"第X条"/编号  <- 确定性判断, 优先于任何概率模型
    ③ 文件名规则     *faq* / *问答* / *条例*
    ④ 通用兜底       递归分隔符 + 定长窗口

刻意不做的: 用 LLM 给文档分类。
    "这段文本里有没有『第X条』" 是正则能 100% 确定的事, 让概率模型去猜
    只会引入不确定性, 还多一次 API 调用。LLM 分类的位置留作 hook(见 classify_by_llm),
    仅在 ①②③ 全部无法判定时才启用。
"""
from __future__ import annotations

import re
from typing import Any

from app.core.ingest.base import BaseIngestor, IngestResult
from app.core.ingest.faq import FaqIngestor
from app.core.ingest.generic import GenericIngestor
from app.core.ingest.statute import StatuteIngestor

# 判定来源(会写入 documents.doc_type_source, 用于质量归因)
SOURCE_USER = "user"
SOURCE_DETECTOR = "detector"
SOURCE_FILENAME = "filename"
SOURCE_GENERIC = "generic"

# 探测器得分达到这个阈值才认领
DETECTOR_MIN_SCORE = 0.5

REGISTRY: dict[str, BaseIngestor] = {}

_FILENAME_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(faq|问答|常见问题|答疑)", re.IGNORECASE), "faq"),
    (re.compile(r"(中华人民共和国|条例|司法解释|实施细则|办法$)"), "statute"),
]


def register(ingestor: BaseIngestor, *, override: bool = False) -> None:
    """注册一个摄取器。新增文档类型只需: 实现子类 + 在这里注册。"""
    if not ingestor.doc_type:
        raise ValueError("摄取器必须定义 doc_type")
    if ingestor.doc_type in REGISTRY and not override:
        raise ValueError(f"doc_type 已注册: {ingestor.doc_type}")
    REGISTRY[ingestor.doc_type] = ingestor


def get_ingestor(doc_type: str) -> BaseIngestor | None:
    return REGISTRY.get(doc_type)


def resolve_ingestor(
    text: str,
    filename: str = "",
    user_hint: str | None = None,
) -> tuple[BaseIngestor, str, dict[str, Any]]:
    """选出一个摄取器。

    Returns:
        (ingestor, source, detail)
        detail 里带各探测器的得分, 便于事后解释"为什么走了这条策略"。
    """
    text = text or ""
    detail: dict[str, Any] = {"scores": {}}

    # ① 用户指定
    if user_hint:
        ingestor = REGISTRY.get(user_hint)
        if ingestor is not None:
            detail["reason"] = "user_hint"
            return ingestor, SOURCE_USER, detail
        detail["reason"] = f"user_hint_not_found:{user_hint}"

    # ② 结构探测器
    best: BaseIngestor | None = None
    best_score = 0.0
    for doc_type, ingestor in REGISTRY.items():
        if doc_type == GenericIngestor.doc_type:
            continue
        try:
            score = ingestor.detect(text, filename)
        except Exception:  # noqa: BLE001 - 单个探测器出错不应影响路由
            score = 0.0
        detail["scores"][doc_type] = round(score, 3)
        if score > best_score:
            best_score, best = score, ingestor

    if best is not None and best_score >= DETECTOR_MIN_SCORE:
        detail["reason"] = f"structure_detector:{best_score:.3f}"
        return best, SOURCE_DETECTOR, detail

    # ③ 文件名规则
    for pattern, doc_type in _FILENAME_RULES:
        if pattern.search(filename or ""):
            ingestor = REGISTRY.get(doc_type)
            if ingestor is not None:
                detail["reason"] = f"filename_rule:{pattern.pattern}"
                return ingestor, SOURCE_FILENAME, detail

    # ④ 通用兜底
    detail["reason"] = "generic_fallback"
    return REGISTRY[GenericIngestor.doc_type], SOURCE_GENERIC, detail


def classify_by_llm(text: str, filename: str = "") -> str | None:
    """LLM 分类 hook(未启用)。

    只有在 ①②③ 全部无法判定时才值得启用 —— 到那时用 LLM 读前 N 行判断类型。
    当前保留占位, 是为了让"判定来源"这条链路完整(doc_type_source 支持 'llm')。
    """
    return None


def ingest_text(
    text: str,
    *,
    filename: str = "",
    user_hint: str | None = None,
    doc_meta: dict[str, Any] | None = None,
) -> tuple[IngestResult, str, dict[str, Any]]:
    """一站式入口: 路由 + 解析 + 自动降级。

    若选中的摄取器解析不出任何片段(结构探测误判、或文档格式异常),
    自动降级到通用摄取器, 并保留"曾尝试过的策略"信息。
    """
    ingestor, source, detail = resolve_ingestor(text, filename, user_hint)
    result = ingestor.parse(text, doc_meta or {})

    if result.sections:
        return result, source, detail

    # 降级: 选中的摄取器没吐出内容
    fallback = REGISTRY.get(GenericIngestor.doc_type)
    if fallback is not None and ingestor.doc_type != fallback.doc_type:
        fallback_result = fallback.parse(text, doc_meta or {})
        if fallback_result.sections:
            fallback_result.detail["fallback_from"] = ingestor.doc_type
            fallback_result.detail["fallback_reason"] = detail.get("reason")
            return fallback_result, SOURCE_GENERIC, detail

    return result, source, detail


def _register_builtins() -> None:
    register(StatuteIngestor())
    register(FaqIngestor())
    register(GenericIngestor())


_register_builtins()
