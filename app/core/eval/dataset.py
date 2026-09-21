"""
评测数据集: 加载与自检

评测集写错了比没有评测集更糟 —— 会出现"怎么优化都上不去"的假象,
而实际原因是标准答案标错了。所以自检是必做步骤, 不是可选项。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_DATASET = Path("data/eval/legal_qa_eval.jsonl")

VALID_KINDS = {"term", "colloquial", "compound", "out_of_scope", "validity"}


@dataclass
class EvalCase:
    id: str
    question: str
    kind: str
    answerable: bool
    required: List[str] = field(default_factory=list)
    acceptable: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def resolvable(self) -> bool:
        """是否有可判定的标准答案。"""
        return bool(self.required)


def load_dataset(path: str | Path = DEFAULT_DATASET) -> List[EvalCase]:
    """读取 JSONL 评测集。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"评测集不存在: {p}")

    cases: List[EvalCase] = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"评测集第 {line_no} 行 JSON 解析失败: {exc}") from exc
        cases.append(
            EvalCase(
                id=payload["id"],
                question=payload["question"],
                kind=payload.get("kind", "term"),
                answerable=bool(payload.get("answerable", True)),
                required=list(payload.get("required") or []),
                acceptable=list(payload.get("acceptable") or []),
                notes=payload.get("notes", ""),
            )
        )
    return cases


def validate_dataset(
    cases: List[EvalCase],
    corpus_article_nos: Optional[set] = None,
) -> Dict[str, List[str]]:
    """自检评测集。

    Args:
        cases: 评测样本
        corpus_article_nos: 语料里实际存在的条文号; 传入后会校验标准答案
                            是否真的在语料里 —— 这是最容易出错也最难发现的地方
                            (标注时按记忆写"第四十六条", 但语料里根本没有)。
    Returns:
        {"errors": [...], "warnings": [...], "stats": {...}}
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not cases:
        errors.append("评测集为空")
        return {"errors": errors, "warnings": warnings, "stats": {}}

    seen_ids: Dict[str, int] = {}
    seen_questions: Dict[str, str] = {}

    for case in cases:
        if case.id in seen_ids:
            errors.append(f"id 重复: {case.id}")
        seen_ids[case.id] = seen_ids.get(case.id, 0) + 1

        normalized = re.sub(r"\s+", "", case.question)
        if normalized in seen_questions:
            warnings.append(f"问题重复: {case.id} 与 {seen_questions[normalized]}")
        seen_questions[normalized] = case.id

        if not case.question.strip():
            errors.append(f"{case.id}: 问题为空")
        if case.kind not in VALID_KINDS:
            errors.append(f"{case.id}: 未知的 kind={case.kind}")

        if case.answerable and not case.required:
            errors.append(f"{case.id}: 标记为可回答但没有 required 标准答案")
        if not case.answerable and case.required:
            errors.append(f"{case.id}: 标记为不可回答却给了 required 标准答案")

        overlap = set(case.required) & set(case.acceptable)
        if overlap:
            warnings.append(f"{case.id}: 同一条同时出现在 required 与 acceptable: {sorted(overlap)}")

        if corpus_article_nos is not None:
            for article in case.required + case.acceptable:
                if article not in corpus_article_nos:
                    errors.append(f"{case.id}: 标准答案 {article} 在语料中不存在 —— 标注有误")

    stats = {
        "total": len(cases),
        "resolvable": sum(1 for c in cases if c.resolvable),
        "unresolvable": sum(1 for c in cases if not c.resolvable),
        "by_kind": {kind: sum(1 for c in cases if c.kind == kind) for kind in sorted(VALID_KINDS)},
    }
    return {"errors": errors, "warnings": warnings, "stats": stats}
