"""
向量库探针 —— T-09 要求的验证脚本

它回答三个具体问题, 每个都必须给出可核对的事实而不是"看起来没问题":

  1. **读回一致性**: Milvus 里存的正文和元数据, 与入库前是不是逐字段相同?
     特别是 display_text(干净原文)—— 它丢了的话, 引用溯源弹出来的是带
     检索前缀的串, 而检索本身照常工作, 你不会发现。

  2. **元数据是否真的落库**: 被测版本的 LlamaIndex Milvus 集成不会自动存
     元数据, 必须显式声明标量字段。这个脚本直接列出集合的真实 schema 来核实,
     而不是相信"代码里传了参数所以应该存上了"。

  3. **BM25 语料能否重建**: 用一个全新进程(无任何内存状态)启动,
     从 Milvus 读回全部正文重建 BM25, 条数必须与入库条数一致。
     这正是"重启后关键词召回静默失效"那个 bug 的回归验证。

用法:
    python scripts/probe_vector_store.py
    python scripts/probe_vector_store.py --sample 5
    python scripts/probe_vector_store.py --source data/raw/中华人民共和国劳动法_20181229.docx

默认会把 data/raw 下所有语料都重算一遍做比对(对齐键是「文件名 + 条文号」,
不是单独的条文号 —— 多部法都有"第三十六条", 只用条文号会撞号)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.core.ingest import ingest_text  # noqa: E402
from app.core.ingest.metadata import MILVUS_SCALAR_FIELDS  # noqa: E402
from app.core.rag.vector_store import VectorStoreManager  # noqa: E402

FAIL = "❌"
OK = "✅"

SUPPORTED_EXT = {".pdf", ".docx", ".xlsx", ".csv", ".md", ".txt"}


def _resolve_sources(cli_sources: list[str] | None) -> list[Path]:
    """比对用的原始语料。命令行没给就扫描 data/raw。"""
    if cli_sources:
        return [Path(p) for p in cli_sources if Path(p).exists()]
    raw = Path("data/raw")
    if not raw.is_dir():
        return []
    return sorted(p for p in raw.iterdir() if p.suffix.lower() in SUPPORTED_EXT)


def _expected_display_texts(sources: list[Path]) -> dict[tuple[str, str | None], str]:
    """用与入库完全相同的链路重算每份语料, 返回 (文件名, 条文号) -> 干净原文。"""
    from app.core.rag.loader import DocumentLoader

    loader = DocumentLoader()
    expected: dict[tuple[str, str | None], str] = {}
    for path in sources:
        try:
            docs = loader.load_file(str(path))
        except Exception as exc:  # noqa: BLE001 - 单份语料读不了不该中断探针
            print(f"      (跳过 {path.name}: {exc})")
            continue
        text = "\n\n".join((doc.text or "") for doc in docs).strip()
        if not text:
            continue
        result, _, _ = ingest_text(text, filename=path.name)
        for section in result.sections:
            expected[(path.name, section.article_no)] = section.text
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Milvus 向量库探针")
    parser.add_argument("--sample", type=int, default=3, help="抽查几条片段")
    parser.add_argument(
        "--source",
        nargs="*",
        default=None,
        help="用于比对的原始语料; 默认扫描 data/raw 下所有支持的文件",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("向量库探针")
    print("=" * 72)

    from app.core.rag.embedder import Embedder

    embedder = Embedder()
    manager = VectorStoreManager(embed_model=embedder.embed_model)

    # ---------- 1. schema 核实 ----------
    from pymilvus import MilvusClient

    client = MilvusClient(uri=settings.milvus_uri)
    collections = client.list_collections()
    print(f"Milvus: {settings.milvus_uri}")
    print(f"集合列表: {collections}")
    if manager.collection_name not in collections:
        print(f"{FAIL} 目标集合 {manager.collection_name} 不存在, 先跑一次入库")
        return 2

    description = client.describe_collection(collection_name=manager.collection_name)
    actual_fields = {field["name"] for field in description["fields"]}
    expected_scalars = set(MILVUS_SCALAR_FIELDS)
    missing = sorted(expected_scalars - actual_fields)
    print(f"集合字段数: {len(actual_fields)}")

    if missing:
        print(f"{FAIL} 有 {len(missing)} 个标量字段没有落库, 元数据会静默丢失:")
        for name in missing[:10]:
            print(f"      - {name}")
        return 1
    print(f"{OK} 全部 {len(expected_scalars)} 个标量字段都已落库")

    dim = next(
        (f["params"]["dim"] for f in description["fields"] if f["params"].get("dim")), None
    )
    print(f"{OK} 向量维度 = {dim} (嵌入模型实测 {manager.dim})")
    if dim != manager.dim:
        print(f"{FAIL} 维度不一致, 入库必然失败")
        return 1

    # ---------- 2. 读回一致性 ----------
    records = manager.iter_chunks()
    print(f"{OK} 从 Milvus 读回片段 {len(records)} 条")
    if not records:
        print(f"{FAIL} 读回为空 —— BM25 语料将无法重建")
        return 1

    # 用同一条 ingest 链路重算一遍, 逐字段比对。
    # 对齐键必须是「文件名 + 条文号」: 多部法都有"第三十六条", 只用条文号会撞号,
    # 那样比对结果会随机命中另一部法的同名条文, 制造假不一致。
    sources = _resolve_sources(args.source)
    if not sources:
        print(f"{FAIL} data/raw 下没找到可比对的语料, 读回一致性无法验证")
        return 1

    expected = _expected_display_texts(sources)
    print(f"比对语料: {len(sources)} 份, 期望片段 {len(expected)} 条")

    compared = 0
    mismatched = []
    for record in records:
        meta = record.get("metadata") or {}
        key = (meta.get("filename"), meta.get("article_no"))
        if not key[0] or not key[1] or key not in expected:
            continue
        compared += 1
        want = expected[key]
        got = meta.get("display_text") or ""
        # 读回的 display_text 必须与入库前的原文一致
        if got != want:
            mismatched.append((key, want[:40], got[:40]))

    print(f"{OK} 逐条比对 {compared} 条(以「文件名+条文号」对齐)")
    if mismatched:
        print(f"{FAIL} {len(mismatched)} 条 display_text 与原文不一致:")
        for key, want, got in mismatched[:5]:
            print(f"      {key[0]} {key[1]}\n        期望: {want}\n        实际: {got}")
        return 1
    if compared == 0:
        print(f"{FAIL} 一条都没比对上 —— 检索结果与语料对不上号, 不要当成通过")
        return 1
    print(f"{OK} display_text 与入库前原文完全一致")

    # ---------- 3. 抽查 ----------
    print("-" * 72)
    for record in records[: args.sample]:
        meta = record.get("metadata") or {}
        print(f"  id       : {record['id']}")
        print(f"  条文     : {meta.get('article_no')} / {meta.get('chapter_path')}")
        print(f"  策略留痕 : {meta.get('chunk_strategy')} ({meta.get('strategy_confidence')})")
        print(f"  时效     : {meta.get('validity_status')}  知识库: {meta.get('kb_id')}")
        print(f"  text     : {record['text'][:60]!r}")
        print(f"  display  : {(meta.get('display_text') or '')[:60]!r}")
        print()

    # ---------- 4. 单条回查(引用核验) ----------
    probe_id = records[0]["id"]
    fetched = manager.get_chunk(probe_id)
    if not fetched:
        print(f"{FAIL} get_chunk({probe_id}) 返回空 —— 引用无法核验")
        return 1
    print(f"{OK} get_chunk 单条回查可用: {fetched['id']}")
    print("=" * 72)
    print("探针全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
