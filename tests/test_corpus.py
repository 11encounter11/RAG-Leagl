# -*- coding: utf-8 -*-
"""语料一致性：`data/raw/` 下的文件必须与 `data/corpus_manifest.json` 一致。

这段断言原来只在 `scripts/verify_corpus.py` 里，**靠手工运行**。
搬进 pytest 之后，`pytest -q` 会顺带校验语料 —— 语料被换掉/改坏时会当场发现，
而不是等到某次评测跑出奇怪数字才回头查。

校验三件事：
  1. 清单里列的文件都在（缺失 → 失败）
  2. `data/raw` 里没有清单外的文件（多余 → 失败：可能是错下的版本）
  3. 每份文件的内容哈希与清单一致（被替换/改动 → 失败）

注：`dates_verified == false` 只作提示，不算失败 —— 它表示"日期尚未二次确认"，
是已知状态（见 `08-语料清单.md` §五）。
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "corpus_manifest.json"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.exists(), f"语料清单不存在: {MANIFEST}"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_raw_dir_exists():
    assert RAW_DIR.is_dir(), f"语料目录不存在: {RAW_DIR}（语料不入库，需按 08 清单重新下载）"


def test_all_listed_files_present(manifest):
    missing = [e["file"] for e in manifest["files"] if not (RAW_DIR / e["file"]).exists()]
    assert missing == [], f"清单里有 {len(missing)} 份文件在 data/raw 找不到: {missing}"


def test_no_unlisted_files(manifest):
    listed = {e["file"] for e in manifest["files"]}
    actual = {p.name for p in RAW_DIR.iterdir() if p.is_file()}
    extra = sorted(actual - listed)
    assert extra == [], (
        f"data/raw 里有 {len(extra)} 份文件不在清单中: {extra}\n"
        "若是有意新增的语料，请补进 corpus_manifest.json；否则可能是错下的版本。"
    )


def test_file_hashes_match(manifest):
    mismatched = []
    for entry in manifest["files"]:
        path = RAW_DIR / entry["file"]
        if not path.exists():
            continue
        actual = _sha256(path)
        if entry.get("sha256") != actual:
            mismatched.append(entry["file"])
    assert mismatched == [], f"{len(mismatched)} 份文件内容与清单不符: {mismatched}"


def test_manifest_declares_version_pairs(manifest):
    """P1 的时效过滤要有"同一部法多版本"才能验证。

    这条测试锁住这个前提：如果语料被换成每部法只有一个版本，
    T-28（默认不召回旧版 / 关掉开关才能召回）就没法验证了。
    """
    versions_per_law: dict[str, int] = {}
    for entry in manifest["files"]:
        versions_per_law[entry["law_name"]] = versions_per_law.get(entry["law_name"], 0) + 1
    pairs = {law: n for law, n in versions_per_law.items() if n >= 2}
    assert len(pairs) >= 2, (
        f"至少需要 2 部法各有两个版本才能验证时效过滤，当前只有: {pairs}"
    )
