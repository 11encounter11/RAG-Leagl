"""
模块健康检查 —— 逐个 import app/ 下的模块，报告失败项。

为什么需要这个脚本：
    静态读代码看不出「导入的类名不存在」这类问题（例如曾把
    SentenceTransformerRerank 误写成 SentenceTransformerReranker，
    又被 try/except 吞掉，导致重排功能从未生效）。
    只有真的 import 一遍才暴露得出来。

用法:
    python scripts/check_imports.py
"""
from __future__ import annotations

import importlib
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def module_name_from_path(path: pathlib.Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def discover_modules() -> list[str]:
    mods = []
    for path in sorted((ROOT / "app").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        mods.append(module_name_from_path(path))
    return mods


def main() -> int:
    mods = discover_modules()
    failures: list[tuple[str, str, str]] = []

    for name in mods:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - 就是要抓全部
            failures.append((name, type(exc).__name__, str(exc)[:300]))

    print(f"共发现 {len(mods)} 个模块，导入失败 {len(failures)} 个")
    print("-" * 70)
    for name, exc_type, msg in failures:
        print(f"[FAIL] {name}")
        print(f"       {exc_type}: {msg}")

    if failures:
        print("-" * 70)
        print("失败详情（最后一个的完整堆栈）:")
        traceback.print_exc()
        return 1

    print("全部模块导入成功 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
