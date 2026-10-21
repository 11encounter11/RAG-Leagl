# -*- coding: utf-8 -*-
"""生成提示词测试(P2-04)。

反思发现的缺口不能只留在日志里: 生成模型必须知道"哪部分没依据", 否则会
拿半套资料给出一个听起来很确定的结论。这些测试钉住提示词里的这一段。
"""
from __future__ import annotations

from app.core.generate.prompt import build_system_prompt


def test_missing_gap_is_added_to_system_prompt():
    message = build_system_prompt(
        "[1] 《劳动法》第二十九条 ……",
        ["劳动合同法第三十九条：严重违纪等过失性辞退例外"],
    )
    content = message["content"]

    assert "检索完整性提示" in content
    assert "劳动合同法第三十九条" in content
    assert "未能核实" in content
    assert "绝对结论" in content


def test_no_missing_gap_keeps_prompt_unchanged():
    message = build_system_prompt("[1] 《劳动法》第二十九条 ……")
    assert "检索完整性提示" not in message["content"]


def test_missing_gap_is_ignored_when_there_is_no_context():
    """没有参考资料时, 提示词本来就会要求不引用法条; 不要再叠加缺口段。"""
    message = build_system_prompt("", ["劳动合同法第三十九条"])
    assert "检索完整性提示" not in message["content"]
