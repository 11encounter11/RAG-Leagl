# -*- coding: utf-8 -*-
"""模块健康检查：app/ 下每个模块都必须能 import 成功。

这段断言原来只存在于 `scripts/check_imports.py`，**靠手工运行**。
而"要靠记性跑的检查"正是这个项目反复栽跟头的地方（见 `03 §九 第 7 条`）——
所以搬进 pytest，跟着 `pytest -q` 自动跑。

写成参数化后，48 个模块各是**一条独立测试**：哪个挂了单独列出来，
而不是像脚本那样"48 个里有一个失败，打印一行"。
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _discover_modules() -> list[str]:
    mods: list[str] = []
    for path in sorted((ROOT / "app").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).with_suffix("")
        mods.append(".".join(rel.parts))
    return mods


MODULES = _discover_modules()


def test_modules_were_discovered():
    """自检：万一目录结构变了导致一个模块都没扫到，这里要拦住，
    否则下面的参数化测试会「零条全部通过」—— 一种静默空跑。"""
    assert len(MODULES) >= 30, f"只扫到 {len(MODULES)} 个模块，检查 app/ 目录结构"


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str):
    importlib.import_module(module_name)
