"""
灌 `law_versions` 种子数据（T-24）

做什么：
    读 `data/corpus_manifest.json`（语料清单），把 9 条版本记录写进 MySQL 的
    `law_versions` 表。**只写版本层**，条文层（`law_articles`）是 T-25 的活。

为什么需要它：
    T-23 建了空表。时效过滤（T-26~T-28）要先有"哪部法有哪几个版本、各自什么时候
    生效"这层数据，否则 `validity_status` 永远是占位值。

幂等性：
    版本 ID 由（法名, 版本标签, 公布日期）确定性推导（见 `app/core/law_versions.py`），
    所以**可以反复执行**：已存在的行会被更新，不会产生重复。

用法：
    python scripts/seed_law_versions.py --dry-run   # 只看会写什么，不落库
    python scripts/seed_law_versions.py             # 写入
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.law_versions import VersionRow, build_version_rows  # noqa: E402
from app.core.memory.db_store import Document, LawVersion, db_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus_manifest.json"

OK = "✅"
FAIL = "❌"
WARN = "⚠️"


def _load_document_ids() -> dict[str, str]:
    """{文件名: documents.id}

    `law_versions.document_id` 指向 `documents.id`，而那个 ID 是入库时随机生成的，
    所以只能靠**文件名**把清单和库里的记录对上。
    """
    session = db_manager.get_session()
    try:
        rows = session.query(Document.id, Document.original_filename).all()
    finally:
        session.close()
    return {name: doc_id for doc_id, name in rows if name}


def _upsert(rows: list[VersionRow]) -> tuple[int, int]:
    """按主键 upsert。返回 (新增, 更新)。"""
    session = db_manager.get_session()
    inserted = updated = 0
    try:
        for row in rows:
            existing = session.get(LawVersion, row.id)
            if existing is None:
                session.add(
                    LawVersion(
                        id=row.id,
                        document_id=row.document_id,
                        law_name=row.law_name,
                        law_level=row.law_level,
                        version_label=row.version_label,
                        publish_date=row.publish_date,
                        effective_date=row.effective_date,
                        validity_status=row.validity_status,
                        superseded_by=row.superseded_by,
                        source_url=row.source_url,
                    )
                )
                inserted += 1
            else:
                existing.document_id = row.document_id
                existing.law_name = row.law_name
                existing.law_level = row.law_level
                existing.version_label = row.version_label
                existing.publish_date = row.publish_date
                existing.effective_date = row.effective_date
                existing.validity_status = row.validity_status
                existing.superseded_by = row.superseded_by
                existing.source_url = row.source_url
                updated += 1
        session.commit()
    finally:
        session.close()
    return inserted, updated


def main() -> int:
    parser = argparse.ArgumentParser(description="灌 law_versions 种子数据")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写库")
    args = parser.parse_args()

    print("=" * 72)
    print("灌 law_versions 种子数据 (T-24)")
    print("=" * 72)

    if not MANIFEST.exists():
        print(f"{FAIL} 找不到语料清单: {MANIFEST}")
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"清单: {MANIFEST.relative_to(ROOT)}（{len(manifest['files'])} 条）")

    document_ids = _load_document_ids()
    print(f"MySQL documents: {len(document_ids)} 行（用于按文件名匹配）")
    print("-" * 72)

    rows, unmatched = build_version_rows(manifest, document_ids)

    # 按法名分组展示，好核对"同一部法有几个版本"
    by_law: dict[str, list[VersionRow]] = {}
    for row in rows:
        by_law.setdefault(row.law_name, []).append(row)

    for law_name, group in by_law.items():
        print(f"  {law_name}（{len(group)} 个版本）")
        for row in sorted(group, key=lambda r: str(r.effective_date)):
            mark = "现行" if row.validity_status == "effective" else row.validity_status
            superseded = " ←被取代" if row.superseded_by else ""
            print(
                f"      {str(row.effective_date):<12} {row.version_label:<12} "
                f"{mark}{superseded}"
            )

    print("-" * 72)
    print(f"待写入: {len(rows)} 条版本")

    if unmatched:
        print(f"{FAIL} 清单里有 {len(unmatched)} 份文件在 MySQL 找不到对应文档：")
        for name in unmatched:
            print(f"      - {name}")
        print("      说明语料还没入库（或入库失败）。请先入库，再跑本脚本。")
        return 1

    if args.dry_run:
        print(f"{OK} dry-run：未写入任何数据")
        return 0

    inserted, updated = _upsert(rows)
    print(f"{OK} 已写入：新增 {inserted} 条，更新 {updated} 条")

    # 写完后立刻自检，避免"以为写进去了其实没有"
    session = db_manager.get_session()
    try:
        total = session.query(LawVersion).count()
        laws = len({r.law_name for r in session.query(LawVersion).all()})
    finally:
        session.close()
    print(f"{OK} 库内现状：law_versions 共 {total} 行，覆盖 {laws} 部法")

    # 表中三列语料清单未提供，明确说明留空的原因
    print(
        f"{WARN} 留空的列：issuing_body / doc_no / expiry_date —— "
        "语料清单未提供，见 docs/08-语料清单.md §五"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
