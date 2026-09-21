"""
文档摄取模块测试

对应验收:
    T-12 ingest 骨架        -> test_detectors / test_registry
    T-13 StatuteIngestor    -> test_statute_*
    T-14 contextual prefix  -> test_contextual_prefix_*
    T-15 元数据补齐          -> test_metadata_*
    T-17 降级链             -> test_fallback_*

设计原则: 测试不依赖 data/raw/(该目录被 gitignore), 全部使用内联样本。
"""
from __future__ import annotations

import pytest

from app.core.ingest import (
    CONFIDENCE_FALLBACK,
    CONFIDENCE_HIGH,
    REQUIRED_CHUNK_FIELDS,
    build_text_nodes,
    ingest_text,
    missing_required_fields,
    resolve_ingestor,
)
from app.core.ingest.detectors import cn_to_int, detect_structure
from app.core.ingest.statute import StatuteIngestor

# ============================================================
# 测试样本
# ============================================================

LAW_TEXT = """中华人民共和国测试法

第一章 总则

第一条 为了测试，制定本法。

第二条 本法适用于测试场景。

第二章 测试程序

第三条 测试应当遵循以下原则：准确性、完整性。

第四条 违反本法规定的，处以罚款。
"""

# 超长法条: 用于验证"单条超过阈值时继续切分, 避免 embedding 静默截断"
LONG_ARTICLE_LAW = (
    "中华人民共和国长条文测试法\n\n第一章 总则\n\n第一条 短条文。\n\n"
    "第二条 "
    + "这是一句用于撑长度的测试内容。" * 200
    + "\n\n第三条 结尾条文。\n"
)

POLICY_TEXT = """员工考勤管理办法

一、总则
本制度适用于公司全体正式员工。

二、考勤时间
工作日上午九点上班，下午六点下班。

三、请假流程
需提前一天在系统内提交申请，由直属主管审批。
"""

FAQ_TEXT = """常见问题

问：试用期最长可以约定多久？
答：劳动合同期限三年以上的，试用期不得超过六个月。

问：加班费怎么计算？
答：工作日加班按工资的 1.5 倍计算。

问：年假有几天？
答：工作满一年不满十年的，年休假五天。
"""

STRUCTURELESS_TEXT = (
    "这是一段没有任何结构的普通文字，既没有法条编号，也没有章节标题，"
    "更没有编号条目。它的存在只是为了验证降级链是否真的能兜住。"
    "我们需要它被切成若干片段，而不是整个上传直接失败。"
) * 12


# ============================================================
# 中文数字与结构探测
# ============================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("四十七", 47),
        ("十", 10),
        ("一", 1),
        ("二十一", 21),
        ("一百零三", 103),
        ("一千二百六十", 1260),
        ("47", 47),
        ("", None),
        ("abc", None),
    ],
)
def test_cn_to_int(raw, expected):
    assert cn_to_int(raw) == expected


def test_detect_structure_on_law_text():
    signals = detect_structure(LAW_TEXT, "测试法.txt")
    assert signals.article_count == 4
    assert signals.heading_count == 2
    assert signals.has_article_structure
    assert signals.law_name == "中华人民共和国测试法"


def test_detect_structure_on_structureless_text():
    signals = detect_structure(STRUCTURELESS_TEXT, "随笔.txt")
    assert not signals.has_article_structure
    assert not signals.has_clause_structure
    assert signals.article_count == 0


# ============================================================
# T-13 按条切分
# ============================================================


def test_statute_parse_splits_by_article():
    result = StatuteIngestor().parse(LAW_TEXT, {"filename": "测试法.txt"})
    assert result.confidence == CONFIDENCE_HIGH
    assert result.strategy == "article_based"
    assert len(result.sections) == 4

    numbers = [s.article_no for s in result.sections]
    assert numbers == ["第一条", "第二条", "第三条", "第四条"]
    assert [s.article_no_norm for s in result.sections] == [1, 2, 3, 4]


def test_statute_article_no_normalized_beyond_ten():
    text = "中华人民共和国测试法\n\n第一章 总则\n\n" + "\n\n".join(
        f"第{n}条 内容{i}。" for i, n in enumerate(
            ["十", "十一", "二十一", "四十七", "一百", "一百零三"], start=1
        )
    )
    result = StatuteIngestor().parse(text)
    assert [s.article_no_norm for s in result.sections] == [10, 11, 21, 47, 100, 103]


def test_statute_chapter_path_is_correct():
    result = StatuteIngestor().parse(LAW_TEXT)
    by_no = {s.article_no: s for s in result.sections}
    assert by_no["第一条"].chapter_path == "第一章 总则"
    assert by_no["第二条"].chapter_path == "第一章 总则"
    assert by_no["第三条"].chapter_path == "第二章 测试程序"
    assert by_no["第四条"].chapter_path == "第二章 测试程序"


def test_statute_body_does_not_swallow_next_chapter_heading():
    """夹在两条法条之间的章节标题不能被算进上一条的正文。"""
    result = StatuteIngestor().parse(LAW_TEXT)
    second = next(s for s in result.sections if s.article_no == "第二条")
    assert "第二章" not in second.text
    assert second.text.startswith("第二条")


def test_statute_returns_degraded_when_no_articles():
    result = StatuteIngestor().parse("这里没有任何法条。")
    assert result.sections == []
    assert not result.structure_detected


def test_long_article_is_split_to_avoid_embedding_truncation():
    result = StatuteIngestor(max_chars=200).parse(LONG_ARTICLE_LAW)
    parts = [s for s in result.sections if s.article_no == "第二条"]
    assert len(parts) > 1
    # 同一条的各个部分共享 article_no, 用 part_index 区分
    assert [p.extra["part_index"] for p in parts] == list(range(len(parts)))
    assert all(p.extra["part_total"] == len(parts) for p in parts)
    assert all(len(p.text) <= 200 for p in parts)


# ============================================================
# T-12 路由
# ============================================================


def test_resolve_law_text_to_statute():
    ingestor, source, detail = resolve_ingestor(LAW_TEXT, "测试法.txt")
    assert ingestor.doc_type == "statute"
    assert source == "detector"
    assert detail["scores"]["statute"] >= 0.5


def test_resolve_respects_user_hint():
    ingestor, source, _ = resolve_ingestor(LAW_TEXT, "测试法.txt", user_hint="faq")
    assert ingestor.doc_type == "faq"
    assert source == "user"


def test_resolve_enterprise_policy_beats_statute_when_many_articles():
    """企业制度也常用"第X条", 但文件名应让它落到 policy。"""
    text = "员工手册\n\n" + "\n\n".join(f"第{n}条 内容。" for n in ["一", "二", "三", "四", "五"])
    ingestor, source, _ = resolve_ingestor(text, "员工手册.txt")
    assert ingestor.doc_type == "policy"
    assert source == "detector"


def test_resolve_structureless_to_generic():
    ingestor, source, _ = resolve_ingestor(STRUCTURELESS_TEXT, "随笔.txt")
    assert ingestor.doc_type == "generic"
    assert source == "generic"


# ============================================================
# T-17 降级链
# ============================================================


def test_fallback_never_returns_empty():
    result, source, _ = ingest_text(STRUCTURELESS_TEXT, filename="随笔.txt")
    assert result.sections, "降级链必须保证永不返回空结果"
    assert source == "generic"
    assert result.confidence in (CONFIDENCE_FALLBACK, "degraded")


def test_fallback_ingestor_used_when_hint_ingestor_yields_nothing():
    """用户说这是 FAQ, 但内容里根本没有问答标记 -> 自动降级到通用切分。"""
    result, source, _ = ingest_text(STRUCTURELESS_TEXT, filename="x.txt", user_hint="faq")
    assert result.sections
    assert source == "generic"
    assert result.detail.get("fallback_from") == "faq"


# ============================================================
# T-14 contextual prefix
# ============================================================


def test_contextual_prefix_injected_into_node_text():
    result, _, _ = ingest_text(LAW_TEXT, filename="测试法.txt")
    nodes = build_text_nodes(result, doc_meta={"tenant_id": "t1", "kb_id": "kb1"})
    third = next(n for n in nodes if n.metadata.get("article_no") == "第三条")
    # 前缀让"章节标题里的词"也能被检索到
    assert "中华人民共和国测试法" in third.text
    assert "第二章 测试程序" in third.text
    assert "第三条" in third.text
    # 原文另外留存, 供引用弹窗展示(不带前缀)
    assert third.metadata["display_text"].startswith("第三条")
    assert "第二章" not in third.metadata["display_text"]


# ============================================================
# T-15 元数据契约
# ============================================================


def test_node_metadata_is_complete():
    result, _, _ = ingest_text(LAW_TEXT, filename="测试法.txt")
    nodes = build_text_nodes(
        result,
        doc_meta={
            "tenant_id": "tenant_a",
            "kb_id": "kb_law",
            "document_id": "doc_1",
            "source_file": "/tmp/测试法.txt",
            "filename": "测试法.txt",
        },
    )
    assert nodes
    for node in nodes:
        assert missing_required_fields(node.metadata) == []
        assert node.metadata["tenant_id"] == "tenant_a"
        assert node.metadata["kb_id"] == "kb_law"
        assert node.metadata["document_id"] == "doc_1"
        assert node.metadata["doc_type"] == "statute"
        assert node.metadata["chunk_strategy"] == "article_based"
        assert node.metadata["strategy_confidence"] == CONFIDENCE_HIGH
        assert node.metadata["structure_detected"] is True
        assert node.metadata["validity_status"] == "effective"
        assert node.metadata["region_code"] == ""


def test_node_metadata_values_are_storable_types():
    """Milvus 标量字段只接受 str/int/float/bool, 不接受 None 或容器。"""
    result, _, _ = ingest_text(LAW_TEXT, filename="测试法.txt")
    nodes = build_text_nodes(result, doc_meta={"tenant_id": "t", "kb_id": "k"})
    for node in nodes:
        for key, value in node.metadata.items():
            assert value is not None, f"{key} 为 None, Milvus 存不进去"
            assert isinstance(value, (str, int, float, bool)), f"{key} 类型非法: {type(value)}"


def test_required_field_list_is_fully_covered():
    """契约里的必填字段必须真的出现在产出里(而不是靠事后占位补空)。"""
    result, _, _ = ingest_text(LAW_TEXT, filename="测试法.txt")
    nodes = build_text_nodes(result, doc_meta={"tenant_id": "t", "kb_id": "k"})
    produced = set(nodes[0].metadata.keys())
    assert set(REQUIRED_CHUNK_FIELDS) - produced == set()
