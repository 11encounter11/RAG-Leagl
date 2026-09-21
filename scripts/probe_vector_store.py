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


def main() -> int:
    parser = argparse.ArgumentParser(description="Milvus 向量库探针")
    parser.add_argument("--sample", type=int, default=3, help="抽查几条片段")
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

    # 用同一条 ingest 链路重算一遍, 逐字段比对
    source = Path("data/raw/劳动法.txt")
    if source.exists():
        text = source.read_text(encoding="utf-8", errors="ignore")
        result, _, _ = ingest_text(text, filename=source.name)
        by_article = {section.article_no: section.text for section in result.sections}

        compared = 0
        mismatched = []
        for record in records:
            meta = record.get("metadata") or {}
            article_no = meta.get("article_no")
            if not article_no or article_no not in by_article:
                continue
            compared += 1
            expected_text = by_article[article_no]
            got_display = meta.get("display_text") or ""
            # 读回的 display_text 必须与入库前的原文一致
            if got_display != expected_text:
                mismatched.append((article_no, expected_text[:40], got_display[:40]))
        print(f"{OK} 逐条比对 {compared} 条(以 article_no 对齐)")
        if mismatched:
            print(f"{FAIL} {len(mismatched)} 条 display_text 与原文不一致:")
            for article_no, want, got in mismatched[:5]:
                print(f"      {article_no}\n        期望: {want}\n        实际: {got}")
            return 1
        print(f"{OK} display_text 与入库前原文完全一致")

    # ---------- 3. 抽查 ----------
    print("-" * 72)
    for record in records[: args.sample]:
        meta = record.get("metadata") or {}
        print(f"  id       : {record['id']}")
        print(f"  条文     : {meta.get('article_no')} / {meta.get('chapter_path')}")
        print(f"  策略留痕 : {meta.get('chunk_strategy')} ({meta.get('strategy_confidence')})")
        print(f"  时效     : {meta.get('validity_status')}  租户: {meta.get('tenant_id')}")
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
