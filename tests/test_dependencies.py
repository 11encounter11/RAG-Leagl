# -*- coding: utf-8 -*-
"""依赖声明审计：代码里 import 的第三方包，必须都写进 `requirements.txt`。

**为什么要有这条测试**（2026-10-14）：

本机是 Anaconda，base 环境自带 pytest / numpy / requests 等一大堆包，
所以"代码用了但没声明"这类问题在本机**永远不会报错** —— 测试全绿，
而按 requirements.txt 装出来的环境跑不起来。这正是 CI 最擅长抓、而我们当时没有的东西。

手工跑一次这个对比，当场发现 6 个包从没声明：
    jieba / numpy / torch / pytest / requests / psutil
其中 jieba 最危险 —— `retriever.py` 里 jieba 缺失时会**静默降级**成字符 bigram
（不报错，只是 BM25 召回质量变差），换台机器也发现不了。

固化成测试后，任何人新增依赖却忘了声明，`pytest` 立刻失败。
"""
from __future__ import annotations

import ast
import importlib.metadata as metadata
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCAN_DIRS = ("app", "scripts", "tests")
LOCAL_ROOTS = {"app", "scripts", "tests", "conftest"}


def _imported_top_level_modules() -> set[str]:
    mods: set[str] = set()
    for dirname in SCAN_DIRS:
        for path in (ROOT / dirname).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mods.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods.add(node.module.split(".")[0])
    return {m for m in mods if m not in sys.stdlib_module_names and m not in LOCAL_ROOTS}


def _declared_distributions() -> set[str]:
    declared: set[str] = set()
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(">=")[0].split("==")[0].split("[")[0].strip()
        declared.add(name.lower().replace("_", "-"))
    return declared


def test_every_imported_package_is_declared():
    declared = _declared_distributions()
    module_to_dist = metadata.packages_distributions()

    undeclared: list[str] = []
    for mod in sorted(_imported_top_level_modules()):
        dists = module_to_dist.get(mod) or []
        if not dists:
            undeclared.append(f"{mod}（该模块当前未安装？）")
            continue
        if not any(d.lower().replace("_", "-") in declared for d in dists):
            undeclared.append(f"{mod} -> {dists[0]}")

    assert undeclared == [], (
        "以下第三方包被代码 import，但没写进 requirements.txt：\n  "
        + "\n  ".join(undeclared)
        + "\n\n本机（Anaconda）可能自带了它们，所以本地跑不会报错，"
        "但按 requirements 装出来的环境会失败。请补进 requirements.txt。"
    )


def test_requirements_declares_test_dependency():
    """测试框架本身也是依赖 —— 缺了会让"按 requirements 装完"的人一条测试都跑不了。"""
    assert "pytest" in _declared_distributions()


def test_scan_actually_found_modules():
    """自检：防止扫描逻辑坏掉后「零个模块 -> 全部通过」这种静默空跑。"""
    mods = _imported_top_level_modules()
    assert len(mods) >= 15, f"只扫到 {len(mods)} 个第三方模块，扫描逻辑可能坏了: {sorted(mods)}"
