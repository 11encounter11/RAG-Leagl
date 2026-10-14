"""法规版本的纯逻辑 —— 从语料清单构造 `law_versions` 行。

为什么单独成模块（而不是全塞进脚本）：
  1. 这段是**纯逻辑**（输入清单 → 输出行），可以脱网单测；
  2. 灌种子的脚本是**操作**（要写库），按项目分工应留在 `scripts/`；
  3. 后面 T-26 要往 chunk 元数据里写 `version_id`，也会用到同一套"版本标识"规则 ——
     放这里避免两边各写一份。

对应 `docs/04-P1任务清单.md` T-24。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

# 确定性 ID 的命名空间。换掉它会让所有版本 ID 变化（等于重建种子），
# 所以这个常量一旦投产就不该改。
_ID_NAMESPACE = uuid.UUID("6f1b0a4e-0f9a-4a1e-9f0b-2c7d5a3e8b41")


@dataclass(frozen=True)
class VersionRow:
    """`law_versions` 的一行。字段严格对应 `app/core/memory/db_store.py::LawVersion`。"""

    id: str
    document_id: str
    law_name: str
    law_level: str | None
    version_label: str | None
    publish_date: date | None
    effective_date: date | None
    validity_status: str
    superseded_by: str | None
    source_url: str | None

    # 表中还有三列，但语料清单没提供对应信息，刻意留空并在报告里说明：
    #   issuing_body（制定机关）/ doc_no（发文字号）/ expiry_date（失效日期）
    # 详见 `08-语料清单.md` §五。


def make_version_id(law_name: str, version_label: str | None, publish_date: str | None) -> str:
    """由（法名, 版本标签, 公布日期）推出稳定的版本 ID。

    为什么不用随机 UUID：灌种子必须**可重复执行**。随机 ID 会让第二次运行
    插入 9 条重复记录；确定性 ID 则天然幂等（同样的输入 → 同样的主键）。
    """
    key = f"law_version|{law_name}|{version_label or ''}|{publish_date or ''}"
    return str(uuid.uuid5(_ID_NAMESPACE, key))


def _parse_date(value: Any) -> date | None:
    """清单里的日期是 "2009-08-27" 字符串；空值返回 None。"""
    if not value:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _derive_superseded_by(entries: list[dict[str, Any]]) -> dict[str, str | None]:
    """推导"被哪个版本取代"。

    规则：同一部法（`law_name` 相同）里有多个版本时，
    那些**非现行有效**的版本，指向其中唯一 `effective` 的那个。

    清单本身没有这个字段 —— 它是从"哪些版本同属一部法 + 各自的有效状态"推出来的。
    若某部法有多个 effective（数据异常），不猜，一律留空。
    """
    by_law: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_law.setdefault(entry["law_name"], []).append(entry)

    result: dict[str, str | None] = {}
    for law_name, group in by_law.items():
        effectives = [e for e in group if e.get("validity_status") == "effective"]
        if not effectives:
            result[law_name] = None
            continue
        if len(effectives) > 1:
            # 数据异常：一部法在同一批语料里出现多个"现行有效"版本 —— 不猜
            result[law_name] = None
            continue
        result[law_name] = make_version_id(
            law_name, effectives[0].get("version_label"), effectives[0].get("publish_date")
        )

    out: dict[str, str | None] = {}
    for entry in entries:
        vid = make_version_id(
            entry["law_name"], entry.get("version_label"), entry.get("publish_date")
        )
        target = result.get(entry["law_name"])
        # 现行有效的那条自己不指向谁；非现行的指向现行那条
        out[vid] = None if entry.get("validity_status") == "effective" else target
    return out


def build_version_rows(
    manifest: dict[str, Any],
    document_ids: dict[str, str],
) -> tuple[list[VersionRow], list[str]]:
    """把语料清单转成 `law_versions` 行。

    Args:
        manifest: `data/corpus_manifest.json` 的内容
        document_ids: {文件名: documents.id} —— 由调用方从 MySQL 查出。
            `law_versions.document_id` 指向 `documents.id`，而那个 ID 是入库时
            随机生成的，所以只能靠**文件名**把两边对上。

    Returns:
        (rows, unmatched) —— unmatched 是"清单里有、但 MySQL 里找不到对应文档"的文件名。
        这种情况说明语料还没入库（或入库失败），调用方应据此报错而非静默跳过。
    """
    entries: Iterable[dict[str, Any]] = manifest.get("files") or []
    entries = list(entries)
    superseded = _derive_superseded_by(entries)
    source_url = manifest.get("source")

    rows: list[VersionRow] = []
    unmatched: list[str] = []
    for entry in entries:
        filename = entry["file"]
        document_id = document_ids.get(filename)
        if not document_id:
            unmatched.append(filename)
            continue

        vid = make_version_id(
            entry["law_name"], entry.get("version_label"), entry.get("publish_date")
        )
        rows.append(
            VersionRow(
                id=vid,
                document_id=document_id,
                law_name=entry["law_name"],
                law_level=entry.get("law_level"),
                version_label=entry.get("version_label"),
                publish_date=_parse_date(entry.get("publish_date")),
                effective_date=_parse_date(entry.get("effective_date")),
                validity_status=entry.get("validity_status") or "effective",
                superseded_by=superseded.get(vid),
                source_url=source_url,
            )
        )
    return rows, unmatched
