"""
语料校验 —— 核对 data/raw 与 data/corpus_manifest.json 是否一致。

为什么需要它: data/raw 被 .gitignore 忽略(体积大、可重新下载), 语料本身不在版本控制里。
一旦有人重新下载、改了个字、或放错版本, 没有任何东西会报警 —— 而检索质量、
评测指标、时效判断全都建立在这批文件之上。这个脚本就是那道报警。

它检查三件事:
  1. 清单里列的文件是否都在     (缺失 -> 报错)
  2. data/raw 里有没有多余文件  (多余 -> 提示, 因为可能是错下的版本)
  3. 每份文件的内容哈希是否吻合 (被替换/改动 -> 报错)

用法:
    python scripts/verify_corpus.py
    python scripts/verify_corpus.py --update   # 重新下载后, 用当前文件刷新清单里的哈希
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "corpus_manifest.json"

OK = "✅"
FAIL = "❌"
WARN = "⚠️"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"{FAIL} 找不到清单: {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="核对 data/raw 与语料清单")
    parser.add_argument(
        "--update",
        action="store_true",
        help="用当前文件的实际哈希刷新清单(重新下载官方文件后使用)",
    )
    args = parser.parse_args()

    if not RAW_DIR.is_dir():
        raise SystemExit(f"{FAIL} 目录不存在: {RAW_DIR}")

    manifest = load_manifest()
    entries = manifest["files"]

    print("=" * 72)
    print("语料校验")
    print("=" * 72)
    print(f"清单: {MANIFEST.relative_to(ROOT)}  ({len(entries)} 份)")
    print(f"目录: {RAW_DIR.relative_to(ROOT)}")
    print("-" * 72)

    listed = {e["file"] for e in entries}
    actual = {p.name for p in RAW_DIR.iterdir() if p.is_file()}

    missing = sorted(listed - actual)
    extra = sorted(actual - listed)
    mismatched: list[tuple[str, str, str]] = []
    updated = 0

    for entry in entries:
        name = entry["file"]
        path = RAW_DIR / name
        if not path.exists():
            continue
        digest = _sha256(path)
        if args.update:
            if entry.get("sha256") != digest:
                entry["sha256"] = digest
                updated += 1
            continue
        if entry.get("sha256") != digest:
            mismatched.append((name, entry.get("sha256", "")[:16], digest[:16]))

    if args.update:
        manifest["updated_at"] = "auto"
        MANIFEST.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{OK} 清单已刷新, {updated} 条哈希更新")
        return 0

    if missing:
        print(f"{FAIL} 清单里有 {len(missing)} 份文件在 data/raw 找不到:")
        for name in missing:
            print(f"      - {name}")

    if mismatched:
        print(f"{FAIL} {len(mismatched)} 份文件内容与清单不符(被改动或被换了版本):")
        for name, want, got in mismatched:
            print(f"      - {name}\n          清单: {want}...\n          实际: {got}...")

    if extra:
        print(f"{WARN} data/raw 里有 {len(extra)} 份文件不在清单中:")
        for name in extra:
            print(f"      - {name}")
        print("      (若是有意新增的语料, 请补进清单; 否则可能是错下的版本)")

    unverified = [e["file"] for e in entries if not e.get("dates_verified", True)]
    if unverified:
        print(f"{WARN} {len(unverified)} 份的施行日期尚未二次确认:")
        for name in unverified:
            print(f"      - {name}")

    print("-" * 72)
    if missing or mismatched:
        print(f"{FAIL} 校验未通过")
        return 1
    print(f"{OK} 校验通过: {len(listed)} 份语料齐全且内容一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
