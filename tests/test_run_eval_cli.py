# -*- coding: utf-8 -*-
"""评测脚本的命令行契约测试。

存在的原因: `--dataset` 曾经被**静默忽略** —— run_check() 内部写死 load_dataset(),
于是指定了别的数据集, 实际检查的却是默认那份, 照样打印"自检通过"。
这与 N-12(config 注入失效)属同一类错误: 参数看着生效、其实没生效, 且不报错。

所以这里不看"函数返回什么", 直接跑命令行、断言退出码 —— 只有真实入口才算数。
`--check` 不依赖 MySQL/Redis/Milvus, 可以随时跑。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_DATASET = ROOT / "data" / "eval" / "legal_qa_eval.jsonl"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/run_eval.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_check_on_real_dataset_passes():
    result = _run("--check")
    assert result.returncode == 0, result.stdout[-2000:]
    assert "自检通过" in result.stdout


def test_check_honors_dataset_argument(tmp_path):
    """故意造一份错的数据集, 断言 --dataset 被真正使用(退出码非 0)。"""
    rows = [
        json.loads(line)
        for line in REAL_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # 只注入一处错误: 法名指向一部不存在的法
    rows[0]["law"] = "中华人民共和国不存在的法"
    bad = tmp_path / "bad_eval.jsonl"
    bad.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    result = _run("--check", "--dataset", str(bad))

    assert result.returncode == 1, f"错误标注未被拦下:\n{result.stdout[-2000:]}"
    assert "不在语料中" in result.stdout
    assert "自检未通过" in result.stdout
