"""
引用格式与上下文组装的**唯一事实来源**

背景(修复 T-01):
    原实现里 prompt 要求模型用 "[来源: 文件名]" 标注,
    而 context 里给出的却是 "[1] 来源: xxx" —— 两边对不上,
    模型只能自己发明格式, 输出必然混乱, 前端也无法把引用映射回片段。

所以: 引用长什么样, 只在这一个文件里定义。
    - 上下文里给模型的编号格式: 到这里取
    - 系统提示词里对模型的要求:   到这里取
    - 返回给前端的引用结构:       到这里取

分层设计(docs/01-需求文档.md §F3.2):
    P0 阶段 chunk 自带 display_text, 引用可直接展开;
    P1 接入 law_versions / law_articles 后, 这里会改为返回权威条文 ID。
"""
from __future__ import annotations

from typing import Any, Iterable, List

# ---- 编号格式: 只有这一处定义 ----
CITATION_INDEX_FORMAT = "[{index}]"

# ---- 时效状态的中文标签 ----
VALIDITY_LABELS = {
    "effective": "现行有效",
    "amended": "已修订",
    "repealed": "已废止",
    "pending": "尚未生效",
}

# ---- 系统提示词里对引用格式的要求(供 prompt.py 引用) ----
CITATION_INSTRUCTION = (
    "1. 先给结论，再列依据，最后给行动建议；\n"
    "2. **每一条法律依据都必须用 [编号] 标注**，编号与参考资料中的 [编号] 一一对应，"
    "例如“根据《劳动合同法》第四十七条 [1]”；\n"
    "3. 只能使用参考资料中出现的法条，禁止引用任何未在参考资料中出现的法条编号；\n"
    "4. 如果参考资料不足以回答问题，必须明确说明“现有资料中未找到相关依据”，"
    "并说明还需要哪些信息，**不允许编造**；\n"
    "5. 如果用户描述的事实不足以判断，先追问缺失的关键事实（如工作年限、所在城市、"
    "是否签订书面合同等），而不是直接给出结论。"
)


def _meta(node: Any) -> dict:
    return getattr(getattr(node, "node", node), "metadata", {}) or {}


def _text(node: Any) -> str:
    inner = getattr(node, "node", node)
    metadata = getattr(inner, "metadata", {}) or {}
    # display_text 是去掉检索前缀的干净原文(引用溯源展示用)
    return metadata.get("display_text") or getattr(inner, "text", "") or ""


def format_citation_label(metadata: dict, index: int) -> str:
    """生成上下文里那行"来源抬头", 例如:

        [1] 《中华人民共和国劳动合同法》 第四章 劳动合同的解除和终止 第四十七条 · 现行有效
    """
    parts: List[str] = [CITATION_INDEX_FORMAT.format(index=index)]

    law_name = metadata.get("law_name") or metadata.get("filename") or "未知来源"
    parts.append(f"《{law_name}》")

    if metadata.get("chapter_path"):
        parts.append(str(metadata["chapter_path"]))
    if metadata.get("section_path"):
        parts.append(str(metadata["section_path"]))
    if metadata.get("article_no"):
        parts.append(str(metadata["article_no"]))
    elif metadata.get("clause_no"):
        parts.append(str(metadata["clause_no"]))
    if metadata.get("question"):
        parts.append(f"问题：{metadata['question']}")

    # Excel/CSV 类来源保留行列信息
    if metadata.get("sheet"):
        parts.append(f"[{metadata['sheet']}表]")
    if metadata.get("row"):
        parts.append(f"第{metadata['row']}行")

    label = " ".join(parts)

    validity = metadata.get("validity_status")
    if validity:
        label += f" · {VALIDITY_LABELS.get(validity, validity)}"
    if metadata.get("part_total", 1) and int(metadata.get("part_total") or 1) > 1:
        label += f"（第{int(metadata.get('part_index', 0)) + 1}/{metadata['part_total']}段）"
    if metadata.get("region_code"):
        label += f" · 地域 {metadata['region_code']}"

    return label


def build_context_block(nodes: Iterable[Any], separator: str = "\n\n---\n\n") -> str:
    """把检索结果拼成给模型的参考资料。

    每条包含: 来源抬头(带编号与时效) + 正文
    """
    blocks: List[str] = []
    for index, node in enumerate(nodes, start=1):
        metadata = _meta(node)
        body = _text(node).strip()
        if not body:
            continue
        blocks.append(f"{format_citation_label(metadata, index)}\n{body}")
    return separator.join(blocks)


def build_sources(nodes: Iterable[Any]) -> List[dict]:
    """构造返回给前端的引用列表(结构见 docs/02-设计文档.md §8.1)。"""
    sources: List[dict] = []
    for index, node in enumerate(nodes, start=1):
        metadata = _meta(node)
        inner = getattr(node, "node", node)
        score = getattr(node, "score", None) or 0.0
        text = _text(node)

        sources.append(
            {
                "index": index,
                # 前端可凭 node_id 调 /api/articles/{node_id} 拉取权威原文
                "node_id": getattr(inner, "node_id", "") or "",
                "document_id": metadata.get("document_id", ""),
                "doc_type": metadata.get("doc_type", ""),
                "law_name": metadata.get("law_name") or metadata.get("filename", "未知来源"),
                "filename": metadata.get("filename", "未知来源"),
                "article_no": metadata.get("article_no", ""),
                "article_no_norm": metadata.get("article_no_norm", 0),
                "chapter_path": metadata.get("chapter_path", ""),
                "clause_no": metadata.get("clause_no", ""),
                "validity_status": metadata.get("validity_status", ""),
                "validity_label": VALIDITY_LABELS.get(
                    metadata.get("validity_status", ""), ""
                ),
                "version_label": metadata.get("version_label", ""),
                "region_code": metadata.get("region_code", ""),
                "sheet": metadata.get("sheet", ""),
                "row": metadata.get("row", ""),
                # 分数: 同时给出"原始分"与"归一化分", 便于线上直接观察阈值该定在哪
                "score_raw": round(float(metadata.get("rerank_raw", 0.0) or 0.0), 4),
                "score_normalized": round(float(score), 4),
                "score": round(float(score), 4),  # 兼容旧字段名
                "search_preview": text[:120] + ("..." if len(text) > 120 else ""),
                "text_preview": text[:120] + ("..." if len(text) > 120 else ""),
                "text": text,
            }
        )
    return sources
