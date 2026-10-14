# -*- coding: utf-8 -*-
"""密钥泄露防线。

起因：`.env` 里存着智谱和 DeepSeek 的真实 API Key。git 历史是**永久的** ——
一旦提交过，后来删掉文件也清不干净（只能重写历史）。所以最好的防线是**每次提交都拦一道**。

覆盖三个层面：
  1. `.env` 有没有被 git 跟踪
  2. `.env` 有没有被 `.gitignore` 忽略
  3. 跟踪的文件与 git 历史里有没有密钥特征

密钥特征刻意写得**窄**（宁可漏报少数变体，也不要天天误报），
两条分别对应本项目在用的两家服务。
"""
from __future__ import annotations

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 刻意收紧：避免误报。若以后接入新服务，在这里补一条。
SECRET_PATTERNS = {
    "OpenAI/DeepSeek 风格 (sk-...)": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "智谱风格 (32位hex.后缀)": re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9]{10,}\b"),
}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, errors="replace"
    )


def test_env_is_not_tracked():
    out = _git("ls-files", ".env")
    assert out.stdout.strip() == "", ".env 被 git 跟踪了 —— 里面的 API Key 会进历史"


def test_env_is_ignored():
    out = _git("check-ignore", ".env")
    assert out.returncode == 0, ".env 没有被 .gitignore 忽略"


def test_no_secret_in_tracked_files():
    files = _git("ls-files").stdout.split()
    hits: list[str] = []
    for name in files:
        path = ROOT / name
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                hits.append(f"{name} <- {label}")
    assert hits == [], f"跟踪的文件里疑似有密钥：{hits}"


def test_no_secret_in_git_history():
    """历史是永久的 —— 提交过一次就得重写历史才能清掉，所以在提交前拦。"""
    diff_text = _git("log", "-p", "--all").stdout
    leaked = [label for label, pattern in SECRET_PATTERNS.items() if pattern.search(diff_text)]
    assert leaked == [], (
        f"git 历史里出现疑似密钥：{leaked}\n"
        "需要重写历史才能清除（`git filter-repo`），并立刻到服务商后台轮换该密钥。"
    )
