# -*- coding: utf-8 -*-
"""`law_versions` 灌种子逻辑（T-24）的单元测试。

测的是**纯逻辑**（清单 → 行），不写库 —— 写库那步是操作，留在
`scripts/seed_law_versions.py` 里手工执行（见 03 §九 第 7 条与"断言 vs 操作"的分工）。

重点覆盖三件容易悄悄错的事：
  1. **版本 ID 必须确定性** —— 否则灌种子不可重复，第二次就插重复行
  2. **superseded_by 必须指对** —— 指向"同法的现行版本"，绝不能跨法指
  3. **数据异常时不猜** —— 一部法出现多个 effective 时留空，而不是硬挑一个
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.core.law_versions import (
    LAW_NAME_ALIASES,
    build_version_rows,
    detect_law_name,
    make_article_id,
    make_version_id,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus_manifest.json"


def _entry(**kw) -> dict:
    base = {
        "file": "x.docx",
        "law_name": "某法",
        "law_level": "法律",
        "version_label": "2020年修正",
        "publish_date": "2020-01-01",
        "effective_date": "2020-06-01",
        "validity_status": "effective",
    }
    base.update(kw)
    return base


# ============================================================
# 版本 ID 的确定性（幂等的前提）
# ============================================================
def test_version_id_is_deterministic():
    a = make_version_id("中华人民共和国劳动法", "2018年修正", "2018-12-29")
    b = make_version_id("中华人民共和国劳动法", "2018年修正", "2018-12-29")
    assert a == b, "同样的输入必须得到同样的 ID，否则灌种子不可重复"


def test_version_id_differs_per_version():
    a = make_version_id("中华人民共和国劳动法", "2009年修正", "2009-08-27")
    b = make_version_id("中华人民共和国劳动法", "2018年修正", "2018-12-29")
    assert a != b, "同一部法的两个版本必须是两个不同的 ID"


def test_version_id_is_stable_across_runs():
    """金标准：把真实值钉死。

    一旦有人改了 ID 生成规则（命名空间常量、key 的拼接方式），这条立刻失败 ——
    因为那等于**全部版本 ID 变化**，已灌的种子会对不上，chunk 里存的 version_id
    也会集体失联。这种"悄无声息重命名主键"的改动必须在提交前拦住。
    """
    assert (
        make_version_id("中华人民共和国劳动法", "2009年修正", "2009-08-27")
        == "3da7140f-b0f1-576c-afcd-65ec3b35db12"
    )
    assert (
        make_version_id("中华人民共和国劳动法", "2018年修正", "2018-12-29")
        == "90ed9065-6212-5eb5-b39d-4938e2a77d76"
    )


# ============================================================
# 基本转换
# ============================================================
def test_build_rows_basic():
    manifest = {"source": "https://example.invalid/", "files": [_entry()]}
    rows, unmatched = build_version_rows(manifest, {"x.docx": "DOC-1"})

    assert unmatched == []
    assert len(rows) == 1
    row = rows[0]
    assert row.document_id == "DOC-1"
    assert row.law_name == "某法"
    assert row.law_level == "法律"
    assert row.publish_date == date(2020, 1, 1)
    assert row.effective_date == date(2020, 6, 1)
    assert row.validity_status == "effective"
    assert row.source_url == "https://example.invalid/"
    assert row.superseded_by is None, "现行有效的那条不指向谁"


def test_document_not_in_db_goes_to_unmatched():
    """清单里有、但库里没有对应文档 —— 必须报出来，不能静默跳过。"""
    manifest = {"files": [_entry(file="a.docx"), _entry(file="b.docx")]}
    rows, unmatched = build_version_rows(manifest, {"a.docx": "DOC-A"})

    assert [r.document_id for r in rows] == ["DOC-A"]
    assert unmatched == ["b.docx"]


def test_missing_dates_become_none():
    manifest = {"files": [_entry(publish_date=None, effective_date=None)]}
    rows, _ = build_version_rows(manifest, {"x.docx": "D"})
    assert rows[0].publish_date is None
    assert rows[0].effective_date is None


# ============================================================
# superseded_by 的推导
# ============================================================
def test_amended_version_points_to_effective_one():
    manifest = {
        "files": [
            _entry(file="old.docx", version_label="2009年修正",
                   publish_date="2009-08-27", validity_status="amended"),
            _entry(file="new.docx", version_label="2018年修正",
                   publish_date="2018-12-29", validity_status="effective"),
        ]
    }
    rows, _ = build_version_rows(manifest, {"old.docx": "D-OLD", "new.docx": "D-NEW"})
    old = next(r for r in rows if r.version_label == "2009年修正")
    new = next(r for r in rows if r.version_label == "2018年修正")

    assert old.superseded_by == new.id, "旧版本必须指向同法的现行版本"
    assert new.superseded_by is None


def test_superseded_by_never_crosses_laws():
    """两法各自成对 —— 关联绝不能跨法。"""
    manifest = {
        "files": [
            _entry(file="a1.docx", law_name="甲法", version_label="v1",
                   publish_date="2010-01-01", validity_status="amended"),
            _entry(file="a2.docx", law_name="甲法", version_label="v2",
                   publish_date="2020-01-01", validity_status="effective"),
            _entry(file="b1.docx", law_name="乙法", version_label="v1",
                   publish_date="2011-01-01", validity_status="amended"),
            _entry(file="b2.docx", law_name="乙法", version_label="v2",
                   publish_date="2021-01-01", validity_status="effective"),
        ]
    }
    ids = {f: f"DOC-{f}" for f in ("a1.docx", "a2.docx", "b1.docx", "b2.docx")}
    rows, _ = build_version_rows(manifest, ids)

    by_id = {r.id: r for r in rows}
    for row in rows:
        if row.superseded_by:
            assert by_id[row.superseded_by].law_name == row.law_name, "跨法指向"


def test_multiple_effective_versions_are_not_guessed():
    """数据异常（一部法两个 effective）时留空，不硬挑一个。"""
    manifest = {
        "files": [
            _entry(file="a.docx", law_name="怪法", version_label="v1", validity_status="effective"),
            _entry(file="b.docx", law_name="怪法", version_label="v2", validity_status="effective"),
        ]
    }
    rows, _ = build_version_rows(manifest, {"a.docx": "A", "b.docx": "B"})
    assert all(r.superseded_by is None for r in rows)


# ============================================================
# 对真实语料清单跑一遍（锁住"7 部法 / 9 条版本 / 2 组配对"这个前提）
# ============================================================
def test_real_manifest_yields_seven_laws_and_nine_versions():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # 用假 document_id：本测试只关心"清单 -> 行"的转换，与库里有没有数据无关
    fake_ids = {e["file"]: f"DOC-{i}" for i, e in enumerate(manifest["files"])}

    rows, unmatched = build_version_rows(manifest, fake_ids)

    assert unmatched == []
    assert len(rows) == 9, "真实清单应有 9 条版本"
    assert len({r.law_name for r in rows}) == 7, "覆盖 7 部法"

    from collections import Counter
    pairs = {law: n for law, n in Counter(r.law_name for r in rows).items() if n >= 2}
    assert len(pairs) >= 2, f"至少要有 2 部法各两版，实际: {pairs}"

    # 每条非现行版本都应指向同法的现行版本
    by_id = {r.id: r for r in rows}
    for row in rows:
        if row.validity_status != "effective":
            assert row.superseded_by, f"{row.law_name} {row.version_label} 没有指向现行版本"
            assert by_id[row.superseded_by].law_name == row.law_name


# ============================================================
# 入库时的版本标识（T-25）
# ============================================================
def test_version_ref_index_maps_filename_to_version():
    manifest = {
        "files": [
            _entry(file="a.docx", law_name="甲法", version_label="v2",
                   publish_date="2020-01-01", effective_date="2020-06-01")
        ]
    }
    from app.core.law_versions import build_version_ref_index

    index = build_version_ref_index(manifest)
    ref = index["a.docx"]

    assert ref.law_name == "甲法"
    assert ref.version_label == "v2"
    assert ref.effective_date == date(2020, 6, 1)
    # 关键：这里算出的 ID 必须与 T-24 灌进 law_versions 的完全一致
    assert ref.version_id == make_version_id("甲法", "v2", "2020-01-01")


def test_article_id_is_deterministic():
    """条文 ID 确定性 → 重新入库时 upsert 而非重复插入。"""
    a = make_article_id("V-1", "第四十七条", 46)
    b = make_article_id("V-1", "第四十七条", 46)
    assert a == b
    assert a != make_article_id("V-1", "第四十八条", 47)


def test_article_id_falls_back_to_seq_without_article_no():
    """FAQ / 降级文本没有条文号，用 seq 兜底，不能撞成一个 ID。"""
    a = make_article_id("V-1", None, 1)
    b = make_article_id("V-1", None, 2)
    assert a != b


# ============================================================
# 问句法名识别（P2-21）
# ============================================================
@pytest.mark.parametrize(
    "query,expected",
    [
        ("劳动法第四十七条讲了什么？", "中华人民共和国劳动法"),
        ("《劳动合同法》第三十九条怎么规定的？", "中华人民共和国劳动合同法"),
        ("社保法关于补缴有什么规定？", "中华人民共和国社会保险法"),
        ("农民工工资支付条例第十七条", "保障农民工工资支付条例"),
        ("带薪年休假条例第六条", "职工带薪年休假条例"),
        ("中华人民共和国劳动争议调解仲裁法第二十七条", "中华人民共和国劳动争议调解仲裁法"),
        # 泛称不是法名：加了过滤会把跨法问题直接搜空
        ("劳动法律法规对竞业限制有什么规定？", None),
        ("劳动法规对加班费怎么规定？", None),
        # 没有法名 / 只有普通词
        ("竞业限制的期限是多久？", None),
        ("合同到期不续签有补偿吗？", None),
        # “劳动法规定……”是法 + 规定，仍然认
        ("劳动法规定每天最多工作几个小时？", "中华人民共和国劳动法"),
        ("", None),
        (None, None),
    ],
)
def test_detect_law_name_is_conservative(query, expected):
    assert detect_law_name(query) == expected


def test_aliases_cover_every_law_in_the_manifest():
    """扩语料后忘了补别名，过滤就永远只认旧法 —— 这条让漂移立刻红。"""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = {entry["law_name"] for entry in manifest["files"]}
    assert names == set(LAW_NAME_ALIASES)


# ============================================================
# 接线测试：`index_file` 是否真的去写 law_articles
#
# 为什么单独写这条：函数各自都对、但"没接上线"是这项目踩过多次的坑
# （N-12 就是 langgraph 的 config 没注入，导致 111 个离线测试全绿而真实请求全挂）。
# 所以这里不看函数，看**装配后的行为**。
# ============================================================
def _fake_sections():
    from app.core.ingest.base import ParsedSection

    return [
        ParsedSection(
            text="第四十七条　经济补偿按劳动者在本单位工作的年限…",
            article_no="第四十七条",
            article_no_norm=47,
            chapter_path="第四章 劳动合同的解除和终止",
            seq=46,
        )
    ]


def test_write_law_articles_fills_version_id_from_manifest(monkeypatch):
    from app.services import rag_service as mod

    captured = {}
    monkeypatch.setattr(
        mod.db_manager, "upsert_law_articles",
        lambda rows: (captured.setdefault("rows", rows), len(rows))[1],
    )
    monkeypatch.setattr(mod.db_manager, "law_version_exists", lambda vid: True)

    result = type("R", (), {"sections": _fake_sections()})()
    count = mod.rag_service._write_law_articles(result, "中华人民共和国劳动法_20181229.docx")

    assert count == 1
    row = captured["rows"][0]
    # 版本必须来自语料清单（按文件名），而不是猜的
    assert row["version_id"] == make_version_id(
        "中华人民共和国劳动法", "2018年修正", "2018-12-29"
    )
    assert row["article_no"] == "第四十七条"
    assert row["article_no_norm"] == 47
    assert row["chapter_path"] == "第四章 劳动合同的解除和终止"
    assert row["content"].startswith("第四十七条")


def test_write_law_articles_with_unknown_file_uses_empty_version(monkeypatch):
    """不在语料清单里的文件（用户临时上传）→ version_id 留空，但条文照样入库。"""
    from app.services import rag_service as mod

    captured = {}
    monkeypatch.setattr(
        mod.db_manager, "upsert_law_articles",
        lambda rows: (captured.setdefault("rows", rows), len(rows))[1],
    )
    monkeypatch.setattr(mod.db_manager, "law_version_exists", lambda vid: False)

    result = type("R", (), {"sections": _fake_sections()})()
    count = mod.rag_service._write_law_articles(result, "临时上传的某文件.txt")

    assert count == 1
    assert captured["rows"][0]["version_id"] == ""


# ============================================================
# chunk 元数据里的版本与时效（T-26）
#
# 在这之前 chunk 的 validity_status 恒为占位值 effective —— 后果是同一部法的
# 两个版本**都被标成"现行有效"**，检索把两个版本一起召回且看不出哪个现行。
# ============================================================
def test_document_meta_carries_real_validity_for_amended_version():
    from app.services import rag_service as mod

    meta = mod.rag_service._document_meta(
        original_filename="中华人民共和国劳动法_20090827.docx",
        kb_id="default",
        document_id="DOC-X",
        source_file="x",
    )

    assert meta["validity_status"] == "amended", "2009 版必须是已修订，而不是占位值 effective"
    assert meta["version_label"] == "2009年修正"
    assert meta["effective_date"] == "2009-08-27"
    assert meta["version_id"] == make_version_id(
        "中华人民共和国劳动法", "2009年修正", "2009-08-27"
    )


def test_document_meta_marks_current_version_effective():
    from app.services import rag_service as mod

    meta = mod.rag_service._document_meta(
        original_filename="中华人民共和国劳动法_20181229.docx",
        kb_id="default",
        document_id="DOC-Y",
        source_file="x",
    )
    assert meta["validity_status"] == "effective"


def test_document_meta_omits_version_for_unlisted_file():
    """不在语料清单里的文件（用户临时上传）不该被硬安一个版本。

    刻意**不写** validity_status 键 —— 让 `build_document_metadata` 的默认值生效，
    避免这里和那边各写一个默认值（两处默认值迟早会不一致）。
    """
    from app.services import rag_service as mod

    meta = mod.rag_service._document_meta(
        original_filename="临时上传.txt",
        kb_id="default",
        document_id="DOC-Z",
        source_file="x",
    )
    assert "version_id" not in meta
    assert "validity_status" not in meta
