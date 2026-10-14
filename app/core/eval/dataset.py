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
    laws: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def resolvable(self) -> bool:
        """是否有可判定的标准答案。"""
        return bool(self.required)

    @property
    def law(self) -> Optional[str]:
        """主法名(展示用)。多条时取第一条。"""
        return self.laws[0] if self.laws else None


def _parse_laws(payload: dict) -> List[str]:
    """`law` 字段允许写成字符串或字符串数组。

    单元素是常态(一个问题通常只需要一部法); 数组用于"这题两部法都算对"的情况,
    例如社保问题既可由《社会保险法》回答、也可由《劳动法》回答。
    """
    raw = payload.get("law", payload.get("laws"))
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise ValueError(f"law 字段格式非法(应为字符串或数组): {raw!r}")


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
                laws=_parse_laws(payload),
                notes=payload.get("notes", ""),
            )
        )
    return cases


def validate_dataset(
    cases: List[EvalCase],
    corpus_index: Optional[Dict[str, set]] = None,
) -> Dict[str, List[str]]:
    """自检评测集。

    Args:
        cases: 评测样本
        corpus_index: {法名: 该法实际存在的条文号集合}; 传入后会校验
                      ① 样本标注的法名确实在语料里
                      ② 标准答案的条文号确实存在于**那一部法**里
                      —— 这是最容易出错也最难发现的地方(标注时按记忆写
                      "第四十六条", 但语料里根本没有; 或者写对了条文号、
                      却记错了它属于哪部法)。
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

        # 法名维度: 语料有多部法时, 只标条文号会产生跨法误判(见 metrics.grade_hit)
        if case.answerable and not case.laws:
            errors.append(f"{case.id}: 可回答但没标注 law(语料含多部法, 条文号会撞号)")
        if not case.answerable and case.laws:
            errors.append(f"{case.id}: 不可回答(超纲)却标注了 law={case.laws}")

        overlap = set(case.required) & set(case.acceptable)
        if overlap:
            warnings.append(f"{case.id}: 同一条同时出现在 required 与 acceptable: {sorted(overlap)}")

        if corpus_index is not None:
            known_laws = sorted(corpus_index)
            for law in case.laws:
                if law not in corpus_index:
                    errors.append(
                        f"{case.id}: 标注的法名《{law}》不在语料中"
                        f"（语料现有: {', '.join(known_laws)}）"
                    )
                    continue
                for article in case.required + case.acceptable:
                    if article not in corpus_index[law]:
                        errors.append(
                            f"{case.id}: 《{law}》里没有 {article} —— 标注有误"
                        )

    stats = {
        "total": len(cases),
        "resolvable": sum(1 for c in cases if c.resolvable),
        "unresolvable": sum(1 for c in cases if not c.resolvable),
        "by_kind": {kind: sum(1 for c in cases if c.kind == kind) for kind in sorted(VALID_KINDS)},
    }
    return {"errors": errors, "warnings": warnings, "stats": stats}
