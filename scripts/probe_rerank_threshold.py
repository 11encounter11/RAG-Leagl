"""重排阈值探针 —— 看"开重排后哪一步把答案弄丢了"。

背景: 2026-10-19 首次开重排跑评测, MRR/NDCG/口语类都涨, 但完整召回与 validity 类在跌。
跌的题分两种机制, 光看评测报告分不出来:

  A. 重排把要的那条排到 top_k 之外(顺序问题);
  B. 阈值闸门把整批候选拒掉(数量问题) —— `SIMILARITY_THRESHOLD=0.1` 是 2026-09-22
     在当时的旧语料(以《劳动法》107 条为主)上按重排分标定的; 语料扩到 703 条后, 绝对分的分布已不可比。
     2026-10-20 (P2-20) 起阈值**只判本批 top1**: 达标就保留、条数交给 top_k,
     所以现在这里只该看到"整批采用"或"整批拒绝", 不再逐条砍。

本探针把两者拆开打印: 全量候选的重排分 + 闸门前后各剩几条 + 标注条文落在第几名。

用法:
    python scripts/probe_rerank_threshold.py                    # 跑评测集里 recall 掉的那几题
    python scripts/probe_rerank_threshold.py --ids L40,L31
    python scripts/probe_rerank_threshold.py --query "自定义问题" --law 中华人民共和国劳动法 --required 第四十七条
    python scripts/probe_rerank_threshold.py --threshold 0      # 看"不套阈值"时长什么样

成本: 每个问句 1 次查询嵌入(不动 703 条语料), **不调 LLM**; 重排器本地加载约 13 秒。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "eval" / "legal_qa_eval.jsonl"

# 2026-10-19 那次评测里 recall / full_recall 掉下去的题, 默认全跑一遍
DEFAULT_IDS = ("L12", "L28", "L29", "L31", "L40")


def load_cases(path: Path) -> dict[str, dict]:
    cases = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            cases[case["id"]] = case
    return cases


def label(node) -> str:
    md = node.node.metadata or {}
    law = (md.get("law_name") or "?").replace("中华人民共和国", "")
    return f"{law}·{md.get('article_no') or md.get('clause_no') or '?'}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default=",".join(DEFAULT_IDS))
    parser.add_argument("--query", help="只跑单个自定义问句")
    parser.add_argument("--law", default="", help="自定义问句的期望法名")
    parser.add_argument("--required", default="", help="自定义问句的期望条文号")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None,
                        help="默认用 settings.SIMILARITY_THRESHOLD; 传 0 表示不套阈值")
    parser.add_argument("--full", action="store_true", help="打印全部候选, 默认只打到期望条文之后")
    args = parser.parse_args()

    settings.ENABLE_RERANK = True
    settings.ENABLE_VALIDITY_FILTER = True
    threshold = settings.SIMILARITY_THRESHOLD if args.threshold is None else args.threshold

    from app.core.rag.retriever import (
        SIMILARITY_THRESHOLD_RULE,
        SIMILARITY_THRESHOLD_SCALE,
        filter_by_similarity_threshold,
    )
    from app.services.rag_service import rag_service

    rag_service.init_index(rebuild_bm25=True, preload_reranker=False)
    retriever = rag_service._retriever
    if retriever is None:
        print("检索器未就绪: 确认 Milvus 已启动且库里有数据")
        return 2
    if not retriever.preload_reranker():
        print("❌ 重排器加载失败 —— 此时分数是 RRF 名次分, 阈值语义完全不同, 本探针结论不成立")
        return 3

    if args.query:
        wanted = [{"id": "自定义", "question": args.query, "law": args.law,
                   "required": [args.required] if args.required else [], "acceptable": []}]
    else:
        cases = load_cases(DEFAULT_DATASET)
        wanted = [cases[i] for i in args.ids.split(",") if i.strip() in cases]

    print(f"阈值={threshold}(尺子={SIMILARITY_THRESHOLD_SCALE}, 规则={SIMILARITY_THRESHOLD_RULE}) | "
          f"top_k={args.top_k} | 时效过滤=开 | 重排=生效\n")

    for case in wanted:
        required = set(case.get("required") or [])
        acceptable = set(case.get("acceptable") or [])
        law = (case.get("law") or "").replace("中华人民共和国", "")
        print("=" * 78)
        print(f"[{case['id']}] {case['question']}")
        print(f"        期望 {law}·{sorted(required)} 可接受 {sorted(acceptable)}")

        fused = retriever.retrieve_raw(case["question"])
        # 生产 retrieve() 是"全量候选送进重排器, 再取前 max(top_k, RERANK_TOP_N) 条"。
        # 这里把 top_n 放宽到全量, 目的是看见期望条文到底被排到第几名 —— 排序本身不变。
        reranked = retriever.rerank(case["question"], fused, top_n=len(fused))
        prod_pool = reranked[: max(args.top_k, settings.RERANK_TOP_N)]
        kept = filter_by_similarity_threshold(
            prod_pool, threshold, scores_are_relevance=retriever.rerank_effective
        )
        # 按对象身份比对: NodeWithScore 是 pydantic 模型, == 会连分数一起比, 拿来判"在不在"
        # 既不直观也没必要。
        kept_ids = {id(n) for n in kept}
        expected_ranks = [
            i + 1
            for i, n in enumerate(reranked)
            if law in label(n) and (n.node.metadata or {}).get("article_no") in required
        ]

        gate_applied = bool(
            retriever.rerank_effective and threshold and threshold > 0
        )
        gate_state = ("未启用" if not gate_applied else ("采用" if kept else "拒绝"))
        print(f"  候选 {len(fused)} 条 -> 重排取前 {len(prod_pool)} 条 -> "
              f"阈值闸门({threshold}, 只判 top1) {gate_state}, 保留 {len(kept)} 条 -> "
              f"本次真正交给模型的条数 = {len(kept[:args.top_k])}")
        if prod_pool:
            print(f"  top1 分数 {prod_pool[0].score:.4f} "
                  f"(尺子={SIMILARITY_THRESHOLD_SCALE}, 阈值={threshold})")
        if expected_ranks:
            print(f"  期望条文重排后位于: 第 {expected_ranks} 名"
                  f"{'（在 top_k 内）' if expected_ranks[0] <= args.top_k else '（掉出 top_k）'}")
        else:
            print("  期望条文根本没进候选池")

        cut = max(args.top_k, expected_ranks[0] if expected_ranks else 0)
        shown = reranked if args.full else reranked[:cut]
        for i, node in enumerate(shown, 1):
            md = node.node.metadata or {}
            art = md.get("article_no") or ""
            mark = ""
            if law and law in label(node):
                if art in required:
                    mark = "  ← 期望(必须命中)"
                elif art in acceptable:
                    mark = "  ← 可接受"
            pool_pos = i <= len(prod_pool)
            if not pool_pos:
                gone = f"  (排在送进闸门的前 {len(prod_pool)} 名之外, 与阈值无关)"
            elif id(node) in kept_ids:
                gone = ""
            else:
                gone = "  ✂ 被阈值闸门整批拒绝"
            print(f"   {i:>3}. {label(node):<26} rel={node.score:.4f} "
                  f"[{md.get('validity_status') or '?'}]{mark}{gone}")
        if not args.full and len(reranked) > len(shown):
            print(f"   ... 其余 {len(reranked) - len(shown)} 条略(--full 看全量)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
