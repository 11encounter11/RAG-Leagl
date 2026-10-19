"""
chunk 元数据契约 —— 唯一事实来源

对应 docs/01-需求文档.md §F3.2 的字段清单。
任何新增切分策略都必须通过本模块生成元数据, 不允许各写各的,
否则"字段齐全"这条验收标准无法被验证。

注意: 缺失元数据的后果是对应能力直接失效 ——
    缺 article_no  -> 无法引用溯源
    缺 document_id -> 无法删除文档(孤儿向量)
    缺 validity    -> 可能引用已废止法条
    缺 region_code -> C 端可能给出错误地域的答案
"""
from __future__ import annotations

from typing import Any

# 必须存在于每条 chunk 上的字段(值为 None 也允许, 但键必须出现)
REQUIRED_CHUNK_FIELDS: tuple[str, ...] = (
    "kb_id",
    "document_id",
    "doc_type",
    "source_file",
    "filename",
    "chunk_index",
    "section_seq",
    "chunk_strategy",
    "strategy_confidence",
    "structure_detected",
    "validity_status",
    # 干净原文（去掉检索前缀）。`build_text_nodes` 对**每一段**都无条件写它，
    # 三类文档类型都有值，所以属于必填而非可选；引用溯源弹窗展示的就是它。
    # 补进契约之后，「契约字段集 == Milvus 标量声明」这个等式才成立 ——
    # tests/test_schema_dictionary.py 靠它拦住"加了列没进契约"这类漂移。
    "display_text",
)

# 结构类字段(并非所有文档类型都有, 例如 FAQ 没有条文号)
OPTIONAL_STRUCTURE_FIELDS: tuple[str, ...] = (
    "article_no",
    "article_no_norm",
    "chapter_path",
    "section_path",
    "clause_no",
    "question",
    "law_name",
    "version_id",
    "version_label",
    "effective_date",
    "expiry_date",
    "region_code",
    "part_index",
    "part_total",
)

VALIDITY_EFFECTIVE = "effective"   # 现行有效
VALIDITY_AMENDED = "amended"       # 已修订
VALIDITY_REPEALED = "repealed"     # 已废止
VALIDITY_PENDING = "pending"       # 尚未生效

# 文档类型
DOC_TYPE_STATUTE = "statute"       # 法律法规 / 司法解释
DOC_TYPE_FAQ = "faq"               # 问答对(自建笔记)
DOC_TYPE_GENERIC = "generic"       # 兜底(无结构文本)


def build_document_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """从上传上下文里抽出"整篇文档级"的元数据, 供每条 chunk 复用。"""
    meta = meta or {}
    return {
        "kb_id": meta.get("kb_id", "default"),
        "document_id": meta.get("document_id", ""),
        "doc_type": meta.get("doc_type", ""),
        "source_file": meta.get("source_file", ""),
        "filename": meta.get("filename", ""),
        "version_id": meta.get("version_id"),
        "version_label": meta.get("version_label"),
        # ---- 时效: P0 先落占位值, P1 接真实法规版本数据 ----
        "effective_date": meta.get("effective_date"),
        "expiry_date": meta.get("expiry_date"),
        "validity_status": meta.get("validity_status", VALIDITY_EFFECTIVE),
        # ---- 地域: 空串表示"全国性/不限定", 用空串而非 None 是为了
        #      Milvus 标量字段对 NULL 支持不佳, 过滤表达式的写法也更统一 ----
        "region_code": meta.get("region_code", ""),
    }


def build_section_extra(
    meta: dict[str, Any] | None,
    *,
    law_name: str | None = None,
    part_index: int | None = None,
    part_total: int | None = None,
) -> dict[str, Any]:
    """组装"本条特有"的元数据。

    刻意不把文档级元数据(知识库/文档ID/时效/地域)放进来。
    理由: 那会让同一份文档的元数据出现两个来源 —— 摄取器在 parse() 时
    捕获一份, 调用方在组装节点时又想注入一份。两者不一致时, 摄取器里的
    旧值会静默覆盖调用方的新值(曾经因此把 kb_id 写成默认值)。
    文档级元数据统一由 nodes.build_text_nodes 注入, 这里只放本条特有的。
    """
    extra: dict[str, Any] = {}
    if law_name:
        extra["law_name"] = law_name
    if part_index is not None:
        extra["part_index"] = part_index
    if part_total is not None:
        extra["part_total"] = part_total
    return extra


def missing_required_fields(metadata: dict[str, Any]) -> list[str]:
    """返回缺失的必填字段(用于测试与入库前自检)。"""
    return [field for field in REQUIRED_CHUNK_FIELDS if field not in metadata]


# ============================================================
# Milvus 标量字段声明
# ============================================================
# 为什么必须显式声明(测试期发现, 记为 N-5):
#   被测版本 llama-index-vector-stores-milvus==1.2.0 的存储布局是
#   `id / doc_id / text / embedding` —— **元数据不会被自动存下来**。
#   如果不声明 scalar_field_names, 那么 article_no / validity_status
#   这些字段在入库后就丢了, 引用溯源与时效过滤全部静默失效
#   (检索还能返回文本, 所以你甚至看不出错了)。
#
# 声明之后还带来一个额外好处: 这些字段成为真实的 Milvus 标量字段,
# 于是"把时效/知识库过滤下推到查询表达式"才有了实现基础。
#
# 类型用字符串表示, 不在契约层 import pymilvus —— 契约不该依赖具体存储实现,
# 由 vector_store.py 负责把字符串映射成 DataType。
MILVUS_SCALAR_FIELDS: dict[str, str] = {
    # ---- 文档级 ----
    "kb_id": "VARCHAR",
    "document_id": "VARCHAR",
    "doc_type": "VARCHAR",
    "filename": "VARCHAR",
    "source_file": "VARCHAR",
    # ---- 切分留痕 ----
    "chunk_strategy": "VARCHAR",
    "strategy_confidence": "VARCHAR",
    "structure_detected": "VARCHAR",
    "chunk_index": "INT64",
    "section_seq": "INT64",
    # ---- 时效与地域(P1 接真实数据, 现在先占位) ----
    "validity_status": "VARCHAR",
    "version_id": "VARCHAR",
    "version_label": "VARCHAR",
    "effective_date": "VARCHAR",
    "expiry_date": "VARCHAR",
    "region_code": "VARCHAR",
    # ---- 结构坐标 ----
    "article_no": "VARCHAR",
    "article_no_norm": "INT64",
    "chapter_path": "VARCHAR",
    "section_path": "VARCHAR",
    "clause_no": "VARCHAR",
    "question": "VARCHAR",
    "law_name": "VARCHAR",
    "part_index": "INT64",
    "part_total": "INT64",
    # ---- 展示用干净原文 ----
    # node.text 里带着检索前缀("《法名》> 章节 > 条文号"), 不能直接拿去展示;
    # 干净的原文必须一起入库, 否则引用溯源弹出来的是带前缀的串。
    "display_text": "VARCHAR",
}

# 这些字段名被 Milvus 的固定 schema 占用, 不能再声明为标量字段
MILVUS_RESERVED_FIELDS: tuple[str, ...] = ("id", "doc_id", "text", "embedding")


# ---- 不进 embedding 的字段 ----
#
# 为什么需要它（2026-10-15 发现）:
#   llama-index 的 `TextNode.get_content(metadata_mode=EMBED)` 默认会把**全部元数据**
#   拼进待嵌入文本。实测后果:
#       node.text  = 72 字  (结构前缀 + 法条正文)
#       实际嵌入   = 577 字  -> 元数据占 88%
#       全体 chunk 平均: 正文只占嵌入内容的 17%
#   于是"决定向量的是元数据, 而不是法条正文"。三个具体害处:
#     1. 同一部法的每条 chunk 都带相同的 law_name / filename / source_file ——
#        这些相同字段占了八成篇幅, 把同法条文的向量拉得极近, 互相更难区分
#        (对应 06 报告里记的"第二十五条与第二十六条向量空间极近")。
#     2. document_id(UUID)/ source_file(本地路径)/ seq / part_index 是纯噪声,
#        对"这条法条讲什么"毫无贡献, 却在稀释正文。
#     3. 与设计意图不符: 02-设计文档 §4.4 写的是"拼接**结构前缀**后再向量化" ——
#        前缀已经在 node.text 里了, 元数据再嵌一遍纯属重复。
#
# 所以: 嵌入内容 = node.text(结构前缀 + 正文), 元数据一律不进。
#
# 取"全部声明字段"而不是手写一份黑名单: 以后往契约里加字段时, 它自动被排除,
# 不会出现"新加的元数据悄悄污染了向量"这种事。
EMBED_EXCLUDED_FIELDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        list(REQUIRED_CHUNK_FIELDS)
        + list(OPTIONAL_STRUCTURE_FIELDS)
        + list(MILVUS_SCALAR_FIELDS)
    )
)


def scalar_default(field_type: str) -> Any:
    """标量字段的默认值。Milvus 的字段默认不可为 NULL, 缺了会被拒。"""
    if field_type == "INT64":
        return 0
    if field_type == "BOOL":
        return False
    if field_type == "DOUBLE":
        return 0.0
    return ""


def ensure_scalar_defaults(metadata: dict[str, Any]) -> dict[str, Any]:
    """补齐并校正所有已声明的 Milvus 标量字段。

    Milvus 插入时要求每个已声明字段都出现且类型匹配, 缺一个整批就失败;
    而我们的元数据里有大量可选字段(FAQ 没有条文号、P0 没有版本号),
    所以必须在入库前统一补齐, 不能指望调用方每次都填全。
    """
    result = dict(metadata)
    for field, field_type in MILVUS_SCALAR_FIELDS.items():
        value = result.get(field)
        if value is None:
            result[field] = scalar_default(field_type)
            continue
        if field_type == "INT64":
            try:
                result[field] = int(value)
            except (TypeError, ValueError):
                result[field] = 0
        elif field_type == "VARCHAR" and not isinstance(value, str):
            result[field] = str(value)
    return result


def scalar_fields_for_milvus() -> tuple[list[str], list[str]]:
    """返回 (字段名列表, 类型名列表), 顺序严格对应。"""
    names = list(MILVUS_SCALAR_FIELDS.keys())
    return names, [MILVUS_SCALAR_FIELDS[name] for name in names]
