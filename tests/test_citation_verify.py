"""
引用编号校验的单元测试(需求 F2.8 / 设计 02 §6.6)。

只覆盖确定性、可离线验证的部分: 回答里出现的 [编号] 必须能对应到本次返回的
sources, 以及该编号指向的条文必须能在权威条文表里查到(查表函数由测试注入,
所以整份文件仍然脱网可跑)。
"""
from app.core.generate.citation import (
    authoritative_article_id,
    extract_citation_indices,
    verify_citation_existence,
    verify_citations,
)
from app.core.law_versions import make_article_id


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


# ============================================================
# 条文存在性校验(T-29): 编号合法, 但那条条文在权威表里查不到
# ============================================================
VERSION = "ver-2018"


def _source(index: int, article_no: str, seq: int, version_id: str = VERSION) -> dict:
    return {
        "index": index,
        "version_id": version_id,
        "article_no": article_no,
        "section_seq": seq,
    }


def _exists_factory(present_ids):
    """造一个查表函数: 只有 present_ids 里的 ID 算存在。"""
    present = set(present_ids)

    def _exists(ids):
        return {i for i in ids if i in present}

    return _exists


def test_authoritative_article_id_is_deterministic():
    """来源算出的主键必须与入库时用的是同一个 —— 否则校验永远查不到。"""
    src = _source(1, "第四十七条", 46)
    assert authoritative_article_id(src) == make_article_id(VERSION, "第四十七条", 46)


def test_authoritative_article_id_none_without_version():
    """没有 version_id(临时上传的文档)算不出主键 —— 这是"无法核验", 不是"不存在"。"""
    assert authoritative_article_id(_source(1, "第一条", 0, version_id="")) is None


def test_existence_passes_when_article_is_in_authority_table():
    sources = [_source(1, "第四十七条", 46)]
    answer = "根据《劳动合同法》第四十七条 [1]，经济补偿按年限计算。"
    exists = _exists_factory([make_article_id(VERSION, "第四十七条", 46)])
    assert verify_citation_existence(answer, sources, exists=exists) == []


def test_existence_flags_citation_missing_from_authority_table():
    """构造一条"编号合法但权威表里没有"的引用 —— 必须被抓出来。"""
    sources = [_source(1, "第三十一条", 30)]
    answer = "依据 [1]，三十日是预告期。"
    exists = _exists_factory([])  # 权威表里什么都没有
    assert verify_citation_existence(answer, sources, exists=exists) == [1]


def test_existence_flags_only_the_missing_one():
    sources = [_source(1, "第一条", 1), _source(2, "第二条", 2)]
    answer = "见 [1] 与 [2]。"
    exists = _exists_factory([make_article_id(VERSION, "第一条", 1)])
    assert verify_citation_existence(answer, sources, exists=exists) == [2]


def test_existence_skips_out_of_range_indices():
    """越界编号由 verify_citations 负责, 这里不重复报 —— 否则同一个问题报两次。"""
    sources = [_source(1, "第一条", 1)]
    exists = _exists_factory([])
    assert verify_citation_existence("依据 [7]。", sources, exists=exists) == []


def test_existence_skips_unverifiable_sources():
    """未登记版本的来源无法核验, 不能判成"引用不存在"。

    否则用户临时上传一份文档, 每条引用都会报错 —— 真问题反而被淹没。
    """
    sources = [_source(1, "第一条", 1, version_id="")]
    exists = _exists_factory([])
    assert verify_citation_existence("依据 [1]。", sources, exists=exists) == []


def test_existence_dedupes_ids_across_citations():
    """多个编号指向同一条条文时, 只查一次; 命中则全部通过。"""
    calls = {"n": 0}
    same = make_article_id(VERSION, "第四十七条", 46)

    def _exists(ids):
        calls["n"] += 1
        assert list(ids) == [same], "同一条条文不该查两次"
        return {same}

    sources = [_source(1, "第四十七条", 46), _source(2, "第四十七条", 46)]
    assert verify_citation_existence("见 [1] 与 [2]。", sources, exists=_exists) == []
    assert calls["n"] == 1


def test_existence_returns_empty_when_no_citation():
    assert verify_citation_existence("现有资料中未找到相关依据。", [], exists=_exists_factory([])) == []


# ============================================================
# 字段对齐: build_sources 与 SourceInfo 是两份声明, 会漂移
# ============================================================
def test_build_sources_fields_match_source_info_model():
    """`build_sources` 的键必须与 `SourceInfo` 的字段完全一致。

    两边各写一份字段清单(一处生成、一处声明给前端), 少一个字段的表现是
    "引用列表里那个字段静默丢失" —— 不报错, 前端只是拿不到值。
    """
    from app.core.generate.citation import build_sources
    from app.models.schemas import SourceInfo

    sources = build_sources([]) or [None]
    if sources == [None]:  # 没有节点时构造一个, 只为拿键集合
        class _Node:
            node_id = "x"
            metadata = {}
            text = "正文"
            score = 0.5

        sources = build_sources([_Node()])

    assert set(sources[0]) == set(SourceInfo.model_fields), (
        "build_sources 与 SourceInfo 字段不一致: "
        f"仅前者有 {set(sources[0]) - set(SourceInfo.model_fields)}, "
        f"仅后者有 {set(SourceInfo.model_fields) - set(sources[0])}"
    )


def test_build_sources_carries_authority_key():
    """T-29 的钥匙: 来源必须带 version_id 与 section_seq, 否则存在性校验无从下手。"""
    from app.core.generate.citation import build_sources

    class _Node:
        node_id = "n1"
        metadata = {
            "version_id": VERSION,
            "article_no": "第四十七条",
            "section_seq": 46,
            "display_text": "第四十七条 经济补偿按劳动者在本单位工作的年限……",
        }
        text = "《中华人民共和国劳动合同法》 > 第四章 > 第四十七条\n第四十七条……"
        score = 0.9

    source = build_sources([_Node()])[0]
    assert authoritative_article_id(source) == make_article_id(VERSION, "第四十七条", 46)
