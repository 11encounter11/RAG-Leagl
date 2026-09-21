"""
检索评测脚本

用法:
    python scripts/run_eval.py --check
        只做评测集自检: 校验标准答案是否真实存在于语料中。
        不依赖 MySQL / Redis / Milvus, 可以随时跑 —— 每次改评测集都该跑一遍。

    python scripts/run_eval.py --top-k 5
        跑完整检索评测, 输出 Recall@k / MRR / NDCG@k, 结果写入
        data/eval/reports/run-<时间戳>.json。需要基础设施可用。

    python scripts/run_eval.py --top-k 5 --mode agentic
        评测 Agentic RAG 工作流(会调用 LLM, 有成本)。用于对比"改写+重排"
        相对纯检索到底有没有增益 —— 没有这个对比, 那些环节就是在白烧钱。

为什么用"条文号匹配"而不是让 LLM 当裁判:
    LLM 裁判自身不稳定, 而且贵。法条编号是确定性的: 检索片段的元数据里
    就带 article_no, 直接比对即可。先把确定性的部分测准, 再谈主观质量。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.eval import (  # noqa: E402
    aggregate,
    grade_of,
    load_dataset,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    report_by_kind,
    validate_dataset,
)
from app.core.eval.metrics import full_recall_at_k  # noqa: E402
from app.core.ingest import ingest_text  # noqa: E402
from app.config.settings import settings  # noqa: E402

REPORT_DIR = Path("data/eval/reports")
CORPUS_DIR = Path("data/raw")


def collect_corpus_article_nos() -> set:
    """把语料跑一遍解析, 收集实际存在的条文号。

    刻意复用生产同一条 ingest 链路 —— 如果这里用另一套正则去数条文,
    就变成了"用一个可能同样有问题的实现去验证另一个实现"。
    """
    article_nos: set = set()
    if not CORPUS_DIR.exists():
        return article_nos
    for path in sorted(CORPUS_DIR.iterdir()):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for doc_type in (None, "statute"):
            result, _, _ = ingest_text(text, filename=path.name, user_hint=doc_type)
            if result.sections:
                for section in result.sections:
                    if section.article_no:
                        article_nos.add(section.article_no)
                break
    return article_nos


def run_check() -> int:
    """评测集自检。"""
    print("=" * 68)
    print("评测集自检")
    print("=" * 68)

    cases = load_dataset()
    corpus_article_nos = collect_corpus_article_nos()
    print(f"语料中的条文号: {len(corpus_article_nos)} 条  (来自 {CORPUS_DIR})")

    outcome = validate_dataset(cases, corpus_article_nos or None)
    stats = outcome["stats"]

    print(f"评测样本: {stats.get('total', 0)} 条")
    print(f"  有标准答案 : {stats.get('resolvable', 0)}")
    print(f"  无标准答案 : {stats.get('unresolvable', 0)}  (超纲题, 正确行为是不召回)")
    print("  按类型     : " + ", ".join(f"{k}={v}" for k, v in (stats.get("by_kind") or {}).items()))

    for warning in outcome["warnings"]:
        print(f"  [警告] {warning}")
    for error in outcome["errors"]:
        print(f"  [错误] {error}")

    if outcome["errors"]:
        print(f"\n自检未通过: {len(outcome['errors'])} 处错误。评测集必须先修正, 否则指标没有意义。")
        return 1
    print("\n自检通过 ✅")
    return 0


async def run_eval(top_k: int, mode: str) -> int:
    """执行检索评测。"""
    from app.services.rag_service import rag_service

    # rebuild_bm25=True 是必须的: 否则 BM25 语料为空, 评出来的其实是
    # "纯向量检索"而不是"混合检索", 而混合检索才是要评测的对象。
    rag_service.init_index(rebuild_bm25=True, preload_reranker=False)
    if rag_service._retriever is None:
        print("检索器未就绪: 请确认 Milvus 已启动且知识库中已有数据。")
        return 2

    print(
        f"检索配置: 重排={'开' if settings.ENABLE_RERANK else '关'} "
        f"Query改写={'开' if settings.ENABLE_QUERY_REWRITE else '关'} "
        f"BM25语料={rag_service._retriever.bm25_size} 条"
    )

    cases = load_dataset()
    per_question = []

    for case in cases:
        if mode == "agentic":
            result = await rag_service.retrieve_with_workflow(case.question, top_k, None)
            nodes = result.get("reranked_docs", [])
            refused = bool(result.get("refuse"))
        else:
            nodes, _ = rag_service.retrieve(case.question, top_k, None)
            refused = not nodes

        grades = [
            grade_of((node.node.metadata or {}).get("article_no", ""), case.required, case.acceptable)
            for node in nodes
        ]
        top_article_nos = [
            (node.node.metadata or {}).get("article_no", "") for node in nodes[:top_k]
        ]

        per_question.append(
            {
                "id": case.id,
                "kind": case.kind,
                "question": case.question,
                "resolvable": case.resolvable,
                "retrieved_articles": top_article_nos,
                "grades": grades,
                "recall": recall_at_k(grades, top_k),
                "full_recall": full_recall_at_k(grades, top_k, len(case.required)),
                "rr": reciprocal_rank(grades),
                "ndcg": ndcg_at_k(grades, top_k),
                "precision": precision_at_k(grades, top_k),
                "refused": 1.0 if refused else 0.0,
                "latency_ms": None,
            }
        )

    overall = aggregate(per_question, top_k)
    by_kind = report_by_kind(per_question, top_k)

    print("=" * 68)
    print(f"检索评测结果  mode={mode}  top_k={top_k}")
    print("=" * 68)
    print(f"样本 {overall['question_count']} 条 (可判定 {overall['resolvable_count']} / 超纲 {overall['unresolvable_count']})")
    print(f"  Recall@{top_k}      : {overall['recall_at_k']:.4f}")
    print(f"  完整召回@{top_k}    : {overall['full_recall_at_k']:.4f}")
    print(f"  MRR              : {overall['mrr']:.4f}")
    print(f"  NDCG@{top_k}       : {overall['ndcg_at_k']:.4f}")
    print(f"  精确率@{top_k}      : {overall['precision_at_k']:.4f}")
    print(f"  超纲题拒答准确率 : {overall['refusal_accuracy']:.4f}")
    print("-" * 68)
    print("分类别:")
    for kind, stats in by_kind.items():
        print(
            f"  {kind:<12} n={stats['question_count']:<3} "
            f"Recall={stats['recall_at_k']:.3f} MRR={stats['mrr']:.3f} NDCG={stats['ndcg_at_k']:.3f}"
        )

    misses = [item for item in per_question if item["resolvable"] and item["recall"] == 0.0]
    if misses:
        print("-" * 68)
        print(f"未命中的问题 ({len(misses)} 条) —— 这是下一步优化的目标清单:")
        for item in misses:
            print(f"  [{item['id']}/{item['kind']}] {item['question'][:40]}")
            print(f"      实际召回: {item['retrieved_articles']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = REPORT_DIR / f"run-{mode}-{stamp}.json"
    report_path.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "mode": mode,
                "top_k": top_k,
                "overall": overall,
                "by_kind": by_kind,
                "per_question": per_question,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("-" * 68)
    print(f"报告已写入: {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="检索评测")
    parser.add_argument("--check", action="store_true", help="只做评测集自检")
    parser.add_argument("--top-k", type=int, default=5, help="评测用的 top_k")
    parser.add_argument("--mode", choices=["direct", "agentic"], default="direct")
    parser.add_argument("--dataset", default="data/eval/legal_qa_eval.jsonl")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if args.dataset != "data/eval/legal_qa_eval.jsonl":
        global REPORT_DIR
        REPORT_DIR = Path(args.dataset).parent / "reports"

    if args.check:
        return run_check()
    return asyncio.run(run_eval(args.top_k, args.mode))


if __name__ == "__main__":
    raise SystemExit(main())
