"""
Milvus 标量字段契约测试

这批测试锁的是"元数据能不能活着到达 Milvus"这件事。
被测版本的 LlamaIndex Milvus 集成不会自动持久化元数据, 所以:
  - 字段没声明  -> 入库即丢, 检索照常返回文本, 故障完全静默;
  - 字段声明了但没值 -> Milvus 拒收整批, 入库直接失败。
两种情况都不是"性能下降", 而是"功能悄悄没了/直接不可用", 必须有测试守住。
"""
from __future__ import annotations

import pytest

from app.core.ingest.metadata import (
    MILVUS_RESERVED_FIELDS,
    MILVUS_SCALAR_FIELDS,
    ensure_scalar_defaults,
    scalar_default,
    scalar_fields_for_milvus,
)


def test_every_declared_field_has_a_known_type():
    known = {"VARCHAR", "INT64", "BOOL", "DOUBLE", "FLOAT"}
    unknown = {name: t for name, t in MILVUS_SCALAR_FIELDS.items() if t not in known}
    assert unknown == {}, f"未知类型会在建集合时炸掉: {unknown}"


def test_no_scalar_field_collides_with_reserved_names():
    """字段名撞上 Milvus 固定 schema 会导致建集合失败, 必须在提交前拦住。"""
    collisions = set(MILVUS_SCALAR_FIELDS) & set(MILVUS_RESERVED_FIELDS)
    assert collisions == set(), f"与保留字段冲突: {collisions}"


def test_display_text_is_declared():
    """回归测试。

    node.text 里带着检索前缀("《法名》> 章节 > 条文号"), 只有 display_text
    是干净原文。它一旦没被声明为标量字段, 入库后就丢了 —— 引用溯源弹出来的
    会变成带前缀的串, 而检索本身完全正常, 所以肉眼很难发现。
    """
    assert "display_text" in MILVUS_SCALAR_FIELDS


def test_article_and_validity_fields_are_declared():
    """引用溯源与时效过滤依赖这几个字段, 缺一个对应能力就静默失效。"""
    for field in ("article_no", "chapter_path", "validity_status", "document_id", "kb_id"):
        assert field in MILVUS_SCALAR_FIELDS, f"{field} 未落库"


def test_names_and_types_stay_aligned():
    names, types = scalar_fields_for_milvus()
    assert len(names) == len(types)
    assert names == list(MILVUS_SCALAR_FIELDS.keys())
    assert types == [MILVUS_SCALAR_FIELDS[n] for n in names]


# ============================================================
# 默认值补齐
# ============================================================
def test_ensure_scalar_defaults_fills_every_declared_field():
    """Milvus 要求每条记录带齐全部已声明字段, 缺一个整批失败。"""
    filled = ensure_scalar_defaults({})
    assert set(MILVUS_SCALAR_FIELDS) <= set(filled)


def test_ensure_scalar_defaults_types_match_declaration():
    filled = ensure_scalar_defaults({})
    for field, field_type in MILVUS_SCALAR_FIELDS.items():
        if field_type == "INT64":
            assert isinstance(filled[field], int) and not isinstance(filled[field], bool)
        elif field_type == "VARCHAR":
            assert isinstance(filled[field], str)


def test_ensure_scalar_defaults_coerces_int64_from_string():
    """元数据里数字常以字符串形式流动, 直接送进 INT64 字段会被 Milvus 拒。"""
    filled = ensure_scalar_defaults({"article_no_norm": "47", "chunk_index": 3})
    assert filled["article_no_norm"] == 47
    assert filled["chunk_index"] == 3


def test_ensure_scalar_defaults_survives_garbage_int64():
    filled = ensure_scalar_defaults({"article_no_norm": "第四十七条"})
    assert filled["article_no_norm"] == 0


def test_ensure_scalar_defaults_stringifies_other_types():
    filled = ensure_scalar_defaults({"law_name": 123})
    assert filled["law_name"] == "123"


def test_ensure_scalar_defaults_preserves_real_values():
    given = {
        "article_no": "第二十一条",
        "chapter_path": "第三章 劳动合同和集体合同",
        "validity_status": "effective",
        "display_text": "第二十一条 劳动合同可以约定试用期。",
    }
    filled = ensure_scalar_defaults(given)
    for key, value in given.items():
        assert filled[key] == value


def test_ensure_scalar_defaults_does_not_mutate_input():
    original = {"article_no": "第一条"}
    ensure_scalar_defaults(original)
    assert original == {"article_no": "第一条"}, "不应就地修改调用方的字典"


def test_scalar_default_values():
    assert scalar_default("VARCHAR") == ""
    assert scalar_default("INT64") == 0
    assert scalar_default("BOOL") is False
    assert scalar_default("DOUBLE") == 0.0


# ============================================================
# 元数据契约与节点的衔接
# ============================================================
def test_nodes_carry_all_scalar_fields_after_sanitize():
    """build_text_nodes 出来的节点必须已经能直接入库(不用调用方再补齐)。"""
    from app.core.ingest import build_text_nodes, ingest_text

    text = "第一条 为了保护劳动者的合法权益，调整劳动关系。\n第二条 在中华人民共和国境内的企业适用本法。"
    result, _, _ = ingest_text(text, filename="测试法.txt", user_hint="statute")
    nodes = build_text_nodes(result, doc_meta={"kb_id": "kb1"})

    assert nodes
    for node in nodes:
        merged = ensure_scalar_defaults(node.metadata)
        assert set(MILVUS_SCALAR_FIELDS) <= set(merged)
        assert merged["kb_id"] == "kb1"
        assert merged["display_text"], "干净原文必须存在, 否则引用展示会带检索前缀"


@pytest.mark.parametrize("field", ["kb_id", "document_id", "display_text"])
def test_critical_fields_appear_on_nodes(field):
    from app.core.ingest import build_text_nodes, ingest_text

    text = "第一条 测试内容。"
    result, _, _ = ingest_text(text, filename="测试法.txt", user_hint="statute")
    nodes = build_text_nodes(result, doc_meta={"document_id": "d"})
    assert field in nodes[0].metadata


def test_document_id_survives_framework_metadata_rewrite():
    """回归(2026-10-14): document_id 会被 llama_index 静默覆盖成字符串 "None"。

    llama_index 的 node_to_metadata_dict() 里有:
        metadata["document_id"] = node.ref_doc_id or "None"
    它是为 Chroma/Pinecone 准备的, 但会**覆盖我们传进去的 document_id**。
    我们若不声明节点的 SOURCE 关系, 这个字段就会变成字符串 "None" ——
    不报错、字段还在, 于是"按 document_id 删向量"永远匹配不到,
    删了文档向量仍留在库里(孤儿向量), 而且没有任何报错。

    这里不测我们自己的封装, 而是直接跑框架那个真正会覆盖的函数 ——
    只有这样才拦得住"框架改了行为"或"我们哪天不再声明关系"。
    """
    from llama_index.core.vector_stores.utils import node_to_metadata_dict

    from app.core.ingest import build_text_nodes, ingest_text

    text = "第一条 测试内容。"
    result, _, _ = ingest_text(text, filename="测试法.txt", user_hint="statute")
    nodes = build_text_nodes(result, doc_meta={"kb_id": "k", "document_id": "DOC-123"})

    assert nodes[0].ref_doc_id == "DOC-123"

    # 走一遍框架真正使用的那个转换, 确认我们的值活下来了
    entry = node_to_metadata_dict(nodes[0], remove_text=True)
    assert entry["document_id"] == "DOC-123", "document_id 被框架覆盖了"


def test_no_source_relation_when_document_id_absent():
    """没有 document_id 时不该硬造一个来源关系。"""
    from app.core.ingest import build_text_nodes, ingest_text

    text = "第一条 测试内容。"
    result, _, _ = ingest_text(text, filename="测试法.txt", user_hint="statute")
    nodes = build_text_nodes(result, doc_meta={"kb_id": "k"})
    assert nodes[0].ref_doc_id is None
