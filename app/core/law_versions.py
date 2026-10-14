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


# ============================================================
# 入库时用：文件名 -> 版本标识
# ============================================================
@dataclass(frozen=True)
class VersionRef:
    """入库时需要的最小版本信息（**不含 document_id**）。

    为什么不含 `document_id`：那要等 `documents` 写进去才有，而入库时我们需要的
    是"这份文件属于哪个版本"—— 那是**入库之前就该知道**的事。
    """

    version_id: str
    law_name: str
    version_label: str | None
    effective_date: date | None
    validity_status: str


def build_version_ref_index(manifest: dict[str, Any]) -> dict[str, VersionRef]:
    """{文件名: 版本标识} —— 入库时按文件名查版本。

    为什么按文件名而不是查数据库（`law_versions.document_id`）：
        - 入库那一刻 `documents` 记录**还没写**，`document_id` 也还不存在；
        - 清库重入时，数据库里更没有可查的东西；
        - 而文件名在入库前后都稳定，且语料清单本身就按文件名登记。

    为什么这样做不会与 `law_versions` 漂移：
        `version_id` 是**确定性**的（由 法名|版本标签|公布日期 推出）。
        从清单算出来的 ID，与 T-24 灌进 `law_versions` 的 ID 必然一致。
    """
    index: dict[str, VersionRef] = {}
    for entry in manifest.get("files") or []:
        vid = make_version_id(
            entry["law_name"], entry.get("version_label"), entry.get("publish_date")
        )
        index[entry["file"]] = VersionRef(
            version_id=vid,
            law_name=entry["law_name"],
            version_label=entry.get("version_label"),
            effective_date=_parse_date(entry.get("effective_date")),
            validity_status=entry.get("validity_status") or "effective",
        )
    return index


def load_version_ref_index(
    manifest_path: str | "Path | None" = None,
) -> dict[str, VersionRef]:
    """读语料清单并建索引。

    清单不存在时返回空字典而**不抛异常** —— 入库不该因为"清单没找到"而整体失败，
    但调用方应据此打警告（条文仍会入库，只是 `version_id` 为空、时效过滤失效）。
    """
    from pathlib import Path as _Path

    if manifest_path is None:
        # app/core/law_versions.py -> 项目根
        path = _Path(__file__).resolve().parents[2] / "data" / "corpus_manifest.json"
    else:
        path = _Path(manifest_path)
    if not path.exists():
        return {}
    import json

    try:
        return build_version_ref_index(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        return {}


def make_article_id(version_id: str | None, article_no: str | None, seq: int) -> str:
    """条文的确定性 ID。

    用确定性 ID 是为了让**重新入库**变成幂等：同一份文件的同一条法条，
    无论入库多少次，主键都一样（配合 upsert，不会产生重复条文）。
    """
    key = f"law_article|{version_id or 'unknown'}|{article_no or seq}"
    return str(uuid.uuid5(_ID_NAMESPACE, key))
