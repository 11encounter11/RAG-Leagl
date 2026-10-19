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
    build_corpus_index,
    DEFAULT_DATASET,
    grade_hit,
    list_laws,
    load_dataset,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    report_by_kind,
    stale_ratio_at_k,
    validate_dataset,
)
from app.core.eval.metrics import full_recall_at_k  # noqa: E402
from app.config.settings import settings  # noqa: E402

REPORT_DIR = Path("data/eval/reports")
CORPUS_DIR = Path("data/raw")


def _short_ref(law_name: str | None, article_no: str | None) -> str:
    """把"法名 + 条文号"缩成便于在报告里读的一小段。

    只打印条文号是不够的 —— 语料含多部法后, 光看"第三十六条"无法判断
    命中的到底是哪一部法, 而复盘失败样本时这恰恰是最关键的信息。
    """
    if not article_no:
        return "(无条文号)"
    if not law_name:
        return article_no
    short = law_name.replace("中华人民共和国", "")
    return f"{short}·{article_no}"


def run_check(dataset: str = str(DEFAULT_DATASET)) -> int:
    """评测集自检。"""
    print("=" * 68)
    print("评测集自检")
    print("=" * 68)

    # 必须用传入的路径 —— 原先这里写死 load_dataset(), 于是 --dataset 被静默忽略:
    # 指定了别的数据集(例如故意造错的自检样例), 实际检查的却是默认那份, 照样打印"通过"。
    cases = load_dataset(dataset)
    print(f"数据集: {dataset}")
    # 按"部"统计, 而不是把所有条文号混成一个集合 —— 混在一起的话,
    # 多部法共享同一批条文号, 标注错误会被并集掩盖(曾经就是这样)。
    corpus_index = build_corpus_index(CORPUS_DIR)
    total_articles = sum(len(v) for v in corpus_index.values())
    print(f"语料: {len(corpus_index)} 部法 / {total_articles} 个条文号  (来自 {CORPUS_DIR})")
    for law in list_laws(corpus_index):
        print(f"    《{law}》 {len(corpus_index[law])} 条")

    if not corpus_index:
        print("[错误] 语料索引为空 —— 自检无法进行(检查 data/raw 下是否真的有语料)")
        return 1

    outcome = validate_dataset(cases, corpus_index)
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


async def run_eval(top_k: int, mode: str, dataset: str = str(DEFAULT_DATASET)) -> int:
    """执行检索评测。"""
    from app.services.rag_service import rag_service

    # rebuild_bm25=True 是必须的: 否则 BM25 语料为空, 评出来的其实是
    # "纯向量检索"而不是"混合检索", 而混合检索才是要评测的对象。
    rag_service.init_index(rebuild_bm25=True, preload_reranker=False)
    if rag_service._retriever is None:
        print("检索器未就绪: 请确认 Milvus 已启动且知识库中已有数据。")
        return 2

    # ---- 重排器: 必须**预加载并确认真的生效**, 不能只看 settings ----
    #
    # 原来这里是 preload_reranker=False, 而报告里记的是 `settings.ENABLE_RERANK`
    # —— 那只是"用户想不想开", 不是"实际开没开"。两者不一致时:
    #   · 分数退回 RRF 名次分量级(第一名 0.03), 相似度阈值会把它全卡掉;
    #   · 而报告上写着 rerank=true, 过几天回头看根本不知道这组数是怎么来的。
    # 加载失败在本机是真实场景(约 0.7 GB 常驻), 所以必须显式区分并大声提示。
    rerank_requested = bool(settings.ENABLE_RERANK)
    rerank_effective = False
    if rerank_requested:
        rerank_effective = bool(rag_service._retriever.preload_reranker())
        if not rerank_effective:
            print(
                "\n" + "=" * 70
                + "\n❌ 配置要求开重排(ENABLE_RERANK=true), 但重排器**没能加载**。\n"
                  "   此时分数是 RRF 名次分而非相关度分, 阈值语义与开重排时完全不同,\n"
                  "   这组数字**不能**与任何'重排生效'的基线比较。\n"
                  "   排查: python scripts/probe_reranker.py (会打印加载耗时与失败原因)\n"
                  "   确实要关重排跑一轮: 显式设 $env:ENABLE_RERANK='false' 再来。\n"
                + "=" * 70
            )
            return 3
    else:
        print(
            "⚠️  本次评测**关闭了重排**: 报告里 rerank_effective=false。\n"
            "    关闭时相似度阈值不生效(见 retriever 的说明), 与开重排的基线不可直接比较。"
        )

    print(
        f"检索配置: 重排={'生效' if rerank_effective else ('配置要求开但加载失败' if rerank_requested else '关')} "
        f"Query改写={'开' if settings.ENABLE_QUERY_REWRITE else '关'} "
        f"时效过滤={'开' if settings.ENABLE_VALIDITY_FILTER else '关'} "
        f"BM25语料={rag_service._retriever.bm25_size} 条"
    )

    cases = load_dataset(dataset)
    per_question = []

    for case in cases:
        if mode == "agentic":
            result = await rag_service.retrieve_with_workflow(case.question, top_k, None)
            nodes = result.get("reranked_docs", [])
            refused = bool(result.get("refuse"))
        else:
            nodes, _ = rag_service.retrieve(case.question, top_k, None)
            refused = not nodes

        # 带上法名分档: 语料含多部法, 只比条文号会把另一部法的同名条文判成命中
        grades = [
            grade_hit(
                (node.node.metadata or {}).get("article_no", ""),
                case.required,
                case.acceptable,
                hit_law=(node.node.metadata or {}).get("law_name") or None,
                expected_laws=case.laws,
            )
            for node in nodes
        ]
        top_article_nos = [
            _short_ref((node.node.metadata or {}).get("law_name"), (node.node.metadata or {}).get("article_no"))
            for node in nodes[:top_k]
        ]
        # 时效过滤的专用观测量: top-k 里混进了多少"非现行有效"片段。
        # 它和召回率是两个独立维度 —— 详见 metrics.stale_ratio_at_k。
        retrieved_status = [
            (node.node.metadata or {}).get("validity_status", "") for node in nodes[:top_k]
        ]

        per_question.append(
            {
                "id": case.id,
                "kind": case.kind,
                "question": case.question,
                "resolvable": case.resolvable,
                "retrieved_articles": top_article_nos,
                "retrieved_status": retrieved_status,
                "grades": grades,
                "recall": recall_at_k(grades, top_k),
                "full_recall": full_recall_at_k(grades, top_k, len(case.required)),
                "rr": reciprocal_rank(grades),
                "ndcg": ndcg_at_k(grades, top_k),
                "precision": precision_at_k(grades, top_k),
                "stale_ratio": stale_ratio_at_k(retrieved_status, top_k),
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
    print(
        f"  旧版混入率@{top_k}  : {overall['stale_ratio_at_k']:.4f}"
        f"   (混入旧版的问题 {overall['questions_with_stale']}/{overall['resolvable_count']})"
    )
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
                # 报告必须自带配置留痕: 否则过两天回头看这份 JSON,
                # 根本分不清它是开过滤还是关过滤跑出来的。
                "validity_filter": settings.ENABLE_VALIDITY_FILTER,
                "rerank": settings.ENABLE_RERANK,
                # 意图与实际分开记: 只记 `rerank`(意图)会让人误以为这组数是
                # 带着 Cross-Encoder 跑出来的。两者不一致时阈值语义完全不同。
                "rerank_requested": rerank_requested,
                "rerank_effective": rerank_effective,
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


async def run_gate_eval(dataset: str = str(DEFAULT_DATASET)) -> int:
    """只评测拒答闸门 —— 不重排、不调 LLM, 因此便宜且可反复跑。

    与 direct/agentic 模式分开统计的理由:
      "该不该拒答"和"检索得准不准"是两个独立能力, 混在一个指标里
      会让"召回到了不该召回的片段"这种错误被平均掉。
    """
    from app.services.rag_service import rag_service

    rag_service.init_index(rebuild_bm25=False, preload_reranker=False)
    if rag_service._retriever is None:
        print("检索器未就绪: 请确认 Milvus 已启动且知识库中已有数据。")
        return 2

    from app.core.workflow.rag_workflow import (
        CONFIDENCE_LOW,
        CONFIDENCE_OUT_OF_DOMAIN,
        classify_domain,
    )

    retriever = rag_service._retriever
    cases = load_dataset(dataset)

    print("=" * 68)
    print("拒答闸门评测 (判据: 向量最高原始相似度, 不含重排/LLM)")
    print("=" * 68)
    print(
        f"阈值: 拒答线={settings.DOMAIN_SCORE_REJECT} "
        f"低置信线={settings.DOMAIN_SCORE_CAUTION}"
    )
    print("-" * 68)

    unresolvable = [c for c in cases if not c.resolvable]
    resolvable = [c for c in cases if c.resolvable]

    refused_out, missed_out, low_out = 0, 0, 0
    refused_in, low_in = 0, 0
    misses = []

    for case in unresolvable:
        score = retriever.vector_top_score(case.question)
        level = classify_domain(score)
        if level == CONFIDENCE_OUT_OF_DOMAIN:
            refused_out += 1
        elif level == CONFIDENCE_LOW:
            low_out += 1
            missed_out += 1
            misses.append((case, score, "低置信(未拒答)"))
        else:
            missed_out += 1
            misses.append((case, score, "放行"))
        print(f"  [超纲] {case.id:<4} {score:.4f}  {level:<14} {case.question[:28]}")

    print("-" * 68)
    for case in resolvable:
        score = retriever.vector_top_score(case.question)
        level = classify_domain(score)
        if level == CONFIDENCE_OUT_OF_DOMAIN:
            refused_in += 1
            misses.append((case, score, "误杀(被拒答)"))
        elif level == CONFIDENCE_LOW:
            low_in += 1

    print(f"超纲题拒答准确率 : {refused_out}/{len(unresolvable)} "
          f"= {refused_out / len(unresolvable):.4f}" if unresolvable else "无超纲样本")
    if low_out:
        print(f"  其中 {low_out} 条只降到低置信而未拒答 (未达硬拒答标准)")
    print(f"库内问题误杀率   : {refused_in}/{len(resolvable)} = "
          f"{refused_in / len(resolvable):.4f}")
    print(f"库内问题低置信率 : {low_in}/{len(resolvable)} = {low_in / len(resolvable):.4f}")

    if misses:
        print("-" * 68)
        print("需要关注的条目:")
        for case, score, what in misses:
            print(f"  [{what}] {case.id} score={score:.4f} {case.question[:32]}")

    report = {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "mode": "gate",
        "thresholds": {
            "reject": settings.DOMAIN_SCORE_REJECT,
            "caution": settings.DOMAIN_SCORE_CAUTION,
        },
        "unresolvable_count": len(unresolvable),
        "resolvable_count": len(resolvable),
        "refusal_accuracy": refused_out / len(unresolvable) if unresolvable else None,
        "false_refusal_rate": refused_in / len(resolvable) if resolvable else None,
        "low_confidence_rate_in_scope": low_in / len(resolvable) if resolvable else None,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"run-gate-{report['timestamp']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 68)
    print(f"报告已写入: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="检索评测")
    parser.add_argument("--check", action="store_true", help="只做评测集自检")
    parser.add_argument("--top-k", type=int, default=5, help="评测用的 top_k")
    parser.add_argument("--mode", choices=["direct", "agentic", "gate"], default="direct")
    parser.add_argument("--dataset", default="data/eval/legal_qa_eval.jsonl")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if args.dataset != "data/eval/legal_qa_eval.jsonl":
        global REPORT_DIR
        REPORT_DIR = Path(args.dataset).parent / "reports"

    if args.check:
        return run_check(args.dataset)
    if args.mode == "gate":
        return asyncio.run(run_gate_eval(args.dataset))
    return asyncio.run(run_eval(args.top_k, args.mode, args.dataset))


if __name__ == "__main__":
    raise SystemExit(main())
