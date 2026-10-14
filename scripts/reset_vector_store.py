"""
清空向量集合 —— 换语料 / 换切分策略 / 换 embedding 模型后重建索引前必须执行。

为什么需要它: 删除 data/raw 下的文件**不等于**删除 Milvus 里的向量。
旧语料(或旧切分策略产生的)片段会继续被检索到, 表现为"明明删掉了还在召回",
而且不会有任何报错。这是本项目最典型的一类静默故障。

本脚本会:
  1. 打印当前集合的实体数(先看清楚要删掉多少)
  2. 要求显式确认(--yes 或交互输入)
  3. drop 掉集合(下次入库时由应用自动按新 schema 重建)

用法:
    python scripts/reset_vector_store.py            # 交互确认
    python scripts/reset_vector_store.py --yes      # 跳过确认(自动化用)
    python scripts/reset_vector_store.py --dry-run  # 只看数量, 不删
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402

OK = "✅"
FAIL = "❌"
WARN = "⚠️"


def main() -> int:
    parser = argparse.ArgumentParser(description="清空 Milvus 向量集合")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认")
    parser.add_argument("--dry-run", action="store_true", help="只统计, 不删除")
    parser.add_argument("--collection", default=None, help="集合名, 默认取 settings")
    args = parser.parse_args()

    from pymilvus import MilvusClient

    collection = args.collection or settings.MILVUS_COLLECTION_NAME

    print("=" * 72)
    print("向量集合重置")
    print("=" * 72)
    print(f"Milvus : {settings.milvus_uri}")
    print(f"集合   : {collection}")

    try:
        client = MilvusClient(uri=settings.milvus_uri)
        collections = client.list_collections()
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 连不上 Milvus —— 先把中间件起起来(docker compose up -d)")
        print(f"      {exc}")
        return 2

    if collection not in collections:
        print(f"{WARN} 集合不存在, 无需清理。当前集合: {collections or '(无)'}")
        return 0

    try:
        stats = client.get_collection_stats(collection_name=collection)
        count = stats.get("row_count", "未知")
    except Exception as exc:  # noqa: BLE001
        count = f"未知 ({exc})"
    print(f"当前实体数: {count}")

    if args.dry_run:
        print(f"{OK} dry-run: 未做任何修改")
        return 0

    if not args.yes:
        print("-" * 72)
        print(f"即将 drop 集合 {collection}，其中全部向量会被删除且不可恢复。")
        print("(语料文件在 data/raw 下, 可重新入库; 向量本身是可再生数据)")
        answer = input(f"确认删除请输入集合名 {collection} : ").strip()
        if answer != collection:
            print("已取消, 未做任何修改")
            return 0

    client.drop_collection(collection_name=collection)
    print(f"{OK} 已删除集合 {collection}")
    print("下一步: 重新入库后, 跑 python scripts/probe_vector_store.py 验证读回一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
