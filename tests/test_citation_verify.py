"""
引用编号校验的单元测试(需求 F2.8 / 设计 02 §6.6)。

只覆盖确定性、可离线验证的部分: 回答里出现的 [编号] 必须能对应到本次返回的
sources。"法条是否真实存在于知识库"需要查 law_articles, 属 P1, 不在本文件范围。
"""
from app.core.generate.citation import extract_citation_indices, verify_citations


def test_extract_indices_from_answer():
    answer = "根据《劳动法》第二十一条 [1]，试用期不得超过六个月 [2]。另见 [10]。"
    assert extract_citation_indices(answer) == [1, 2, 10]


def test_extract_dedupes_and_sorts():
    assert extract_citation_indices("[3] [1] [3]") == [1, 3]


def test_extract_ignores_non_numeric_brackets():
    # 旧格式 [来源: xxx] 不是引用编号, 不能被当成引用
    assert extract_citation_indices("依据 [来源: 劳动合同法] 与 [1]") == [1]


def test_verify_passes_when_all_indices_in_range():
    answer = "结论……依据 [1] 与 [3]。"
    sources = [{"index": 1}, {"index": 2}, {"index": 3}]
    assert verify_citations(answer, sources) == []


def test_verify_detects_out_of_range_index():
    # 只给 2 条来源, 回答却引用 [7] -> 必须被抓出来
    answer = "依据 [1] 与 [7]。"
    sources = [{"index": 1}, {"index": 2}]
    assert verify_citations(answer, sources) == [7]


def test_verify_detects_zero_index():
    assert verify_citations("依据 [0]。", [{"index": 1}]) == [0]


def test_verify_passes_when_answer_has_no_citation():
    assert verify_citations("现有资料中未找到相关依据。", []) == []


def test_verify_flags_any_citation_when_sources_empty():
    assert verify_citations("依据 [1]。", []) == [1]
