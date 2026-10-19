# -*- coding: utf-8 -*-
"""字段字典防漂移测试（把 `docs/13-数据模型与处理链路.md` 钉在代码上）。

**为什么要有这个文件**：这个项目反复出现同一类故障 —— 文档领先或落后于代码，
于是接手的人（以及用户本人）不能信任文档，只能重新读代码验证。
`docs/13` 的目的恰恰是"让用户不用读代码也能看出我改了什么"。
如果它自己会漂移，那个目的就不成立。

所以这里不做"写得漂亮"的功夫，只做一件事：**文档里的字段字典与代码不一致就测试失败。**

比对口径是**集合相等**（不在意顺序），所以文档排版可以调整；
但字段的新增/删除/改名必须同步写进文档，否则红。
"""
from __future__ import annotations

import re
from pathlib import Path

from app.core.ingest.metadata import (
    MILVUS_SCALAR_FIELDS,
    OPTIONAL_STRUCTURE_FIELDS,
    REQUIRED_CHUNK_FIELDS,
)
from app.core.memory.db_store import Base

DOC = Path(__file__).resolve().parents[1] / "docs" / "13-数据模型与处理链路.md"
FENCE = re.compile(r"```schema-dict\n(.*?)```", re.S)


def _dictionary() -> dict[str, list[str]]:
    """解析文档里的字段字典块。"""
    assert DOC.exists(), f"文档不存在: {DOC}"
    match = FENCE.search(DOC.read_text(encoding="utf-8"))
    assert match, "文档里找不到 ```schema-dict 字典块（附录被删了还是改名了？）"

    entries: dict[str, list[str]] = {}
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        # 从**最后一个**冒号切: 键名自身可能带冒号(`milvus:knowledge_base`),
        # 而字段列表里没有冒号。用 partition 会从第一个冒号就切断, 把键名切碎。
        name, fields = line.rsplit(":", 1)
        entries[name.strip()] = [f.strip() for f in fields.split(",") if f.strip()]
    assert entries, "字典块解析结果为空"
    return entries


# 代码侧的真实形状（唯一事实来源）
def _mysql_tables() -> dict[str, set[str]]:
    return {
        table.name: {column.name for column in table.columns}
        for table in Base.metadata.tables.values()
    }


# ============================================================
# 1) MySQL: 文档字典 ↔ ORM
# ============================================================
def test_documented_mysql_columns_match_orm():
    doc = _dictionary()
    orm = _mysql_tables()

    for table, columns in doc.items():
        if table.startswith(("milvus:", "contract:")):
            continue
        assert table in orm, f"文档写了表 `{table}`, 但 ORM 里没有这张表"
        assert set(columns) == orm[table], (
            f"表 `{table}` 字段不一致 —— 文档多: "
            f"{sorted(set(columns) - orm[table])} / 代码多: {sorted(orm[table] - set(columns))}"
        )


def test_every_orm_table_is_documented():
    """新表必须登记。漏登记比写错更危险: 用户看文档会以为这张表不存在。"""
    doc = {k for k in _dictionary() if not k.startswith(("milvus:", "contract:"))}
    missing = set(_mysql_tables()) - doc
    assert missing == set(), f"这些表没写进文档字典: {sorted(missing)}"


def test_documented_columns_have_no_duplicates():
    for table, columns in _dictionary().items():
        assert len(columns) == len(set(columns)), f"`{table}` 字典里有重复字段: {columns}"


# ============================================================
# 2) Milvus 标量列 与 chunk 元数据契约
# ============================================================
def test_documented_milvus_scalars_match_declaration():
    doc = _dictionary()["milvus:knowledge_base"]
    assert set(doc) == set(MILVUS_SCALAR_FIELDS), (
        f"Milvus 标量列不一致 —— 文档多: {sorted(set(doc) - set(MILVUS_SCALAR_FIELDS))} / "
        f"代码多: {sorted(set(MILVUS_SCALAR_FIELDS) - set(doc))}"
    )


def test_documented_contract_matches_metadata_tuples():
    doc = _dictionary()
    assert set(doc["contract:required"]) == set(REQUIRED_CHUNK_FIELDS)
    assert set(doc["contract:optional_structure"]) == set(OPTIONAL_STRUCTURE_FIELDS)


def test_contract_union_equals_milvus_scalars():
    """这条等式是本次新增的核心断言。

    「每条 chunk 该有的元数据键」和「Milvus 实际建的标量列」必须是同一批东西。
    不等就意味着: 要么加列没进契约(字段会静默丢失, 见 test_metadata_contract.py 文件头),
    要么契约里有代码根本不写的键(文档在骗人)。

    2026-10-19 之前这条不成立 —— 差的正是 `display_text`（每条 chunk 都有值、
    引用溯源展示靠它, 却没写进契约元组）。见 docs/13 §4.3。
    """
    contract = set(REQUIRED_CHUNK_FIELDS) | set(OPTIONAL_STRUCTURE_FIELDS)
    declared = set(MILVUS_SCALAR_FIELDS)
    assert contract == declared, (
        f"契约与标量声明不是一套东西 —— 只在契约: {sorted(contract - declared)} / "
        f"只在标量声明: {sorted(declared - contract)}"
    )


def test_required_and_optional_tuples_do_not_overlap():
    """两个元组必须互斥, 否则"必填/可选"的语义就塌了。"""
    overlap = set(REQUIRED_CHUNK_FIELDS) & set(OPTIONAL_STRUCTURE_FIELDS)
    assert overlap == set(), f"同一个键既在必填又在可选: {sorted(overlap)}"
