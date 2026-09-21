"""
拒答闸门阈值标定

为什么需要它:
    "超纲问题该拒答"这条安全底线, 判据是一个数值阈值。阈值拍脑袋定会出两种错:
    定高了把正常问题也拒掉(用户觉得系统没用), 定低了该拒的没拒
    (用劳动法回答商标法问题)。两种错都无法靠"看代码"发现。

    所以: 先在评测集上量出两个分布, 再选点。这个脚本就是那把尺子,
    而且必须可重跑 —— 评测集扩大后阈值要重标。

判据选择:
    用**向量检索的最高原始相似度**, 不用 BM25。理由见实测数据:
    超纲题 L38 的 BM25 得分 41.8, 高于库内问题的中位数 39.9 ——
    BM25 在这个语料上根本分不开领域, 拿它当判据只会制造假安全感。

用法:
    python scripts/calibrate_domain_gate.py
    python scripts/calibrate_domain_gate.py --csv     # 导出逐题明细
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.core.eval import load_dataset  # noqa: E402


def best_threshold(in_scores, out_scores):
    """在候选阈值里选"超纲全拒 + 库内误杀最少"的那个。

    评判顺序刻意如此: 先保证安全(超纲不能漏), 再最小化对正常问题的误伤。
    反过来(先保证不误杀)会让拒答闸门形同虚设。
    """
    candidates = sorted({round(s, 4) for s in list(in_scores) + list(out_scores)})
    best = None
    for t in candidates:
        missed = sum(1 for s in out_scores if s >= t)          # 该拒未拒
        collateral = sum(1 for s in in_scores if s < t)        # 误杀正常问题
        key = (missed, collateral, -t)
        if best is None or key < best[0]:
            best = (key, t, missed, collateral)
    return best[1], best[2], best[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="拒答闸门阈值标定")
    parser.add_argument("--csv", action="store_true", help="导出逐题明细 CSV")
    args = parser.parse_args()

    from app.services.rag_service import rag_service

    rag_service.init_index(rebuild_bm25=False, preload_reranker=False)
    if rag_service._retriever is None:
        print("检索器未就绪: 确认 Milvus 已启动且已有数据")
        return 2
    retriever = rag_service._retriever

    cases = load_dataset()
    rows = []
    for case in cases:
        score = retriever.vector_top_score(case.question)
        rows.append((case.id, case.kind, case.answerable, score, case.question))

    in_scope = [r[3] for r in rows if r[2]]
    out_scope = [r[3] for r in rows if not r[2]]

    print("=" * 74)
    print("拒答闸门阈值标定 —— 判据: 向量检索最高原始相似度")
    print("=" * 74)
    print(f"库内问题 n={len(in_scope):<3} min={min(in_scope):.4f} p25={np.percentile(in_scope,25):.4f} "
          f"中位={np.median(in_scope):.4f} max={max(in_scope):.4f}")
    print(f"超纲问题 n={len(out_scope):<3} min={min(out_scope):.4f} p25={np.percentile(out_scope,25):.4f} "
          f"中位={np.median(out_scope):.4f} max={max(out_scope):.4f}")

    gap_ok = min(in_scope) > max(out_scope)
    print()
    if gap_ok:
        print(f"两个分布可完全分开: 阈值取 ({max(out_scope):.4f}, {min(in_scope):.4f}) 之间即可零误判")
    else:
        print(f"⚠️ 两个分布**区间重叠** [{min(in_scope):.4f}, {max(out_scope):.4f}] ——")
        print("   任何单阈值都必然误判一边。这正是采用两档(硬拒答 + 低置信)的原因。")

    t, missed, collateral = best_threshold(in_scope, out_scope)
    print()
    print(f"最优单阈值 = {t:.4f}")
    print(f"  该拒未拒 : {missed}/{len(out_scope)}")
    print(f"  误杀正常 : {collateral}/{len(in_scope)}")
    if collateral or missed:
        print("  说明: 这个阈值不是免费的, 有代价 —— 报告里要如实写。")

    print()
    print(f"当前配置: ENABLE_DOMAIN_GATE={settings.ENABLE_DOMAIN_GATE} "
          f"REJECT={settings.DOMAIN_SCORE_REJECT} CAUTION={settings.DOMAIN_SCORE_CAUTION}")
    for name, threshold in (("拒答线", settings.DOMAIN_SCORE_REJECT),
                            ("低置信线", settings.DOMAIN_SCORE_CAUTION)):
        hit_out = sum(1 for s in out_scope if s < threshold)
        hit_in = sum(1 for s in in_scope if s < threshold)
        print(f"  {name} {threshold}: 拦住超纲 {hit_out}/{len(out_scope)}, 误杀库内 {hit_in}/{len(in_scope)}")

    print()
    print("超纲题逐条(判据值越低越应该拒):")
    for r in sorted([r for r in rows if not r[2]], key=lambda x: x[3]):
        verdict = "拒答" if r[3] < settings.DOMAIN_SCORE_REJECT else (
            "低置信" if r[3] < settings.DOMAIN_SCORE_CAUTION else "放行 ❌")
        print(f"   {r[0]:<4} {r[3]:.4f}  [{verdict}]  {r[4][:30]}")

    print()
    print("库内问题里判据值最低的 5 条(最可能被误杀):")
    for r in sorted([r for r in rows if r[2]], key=lambda x: x[3])[:5]:
        verdict = "拒答 ❌" if r[3] < settings.DOMAIN_SCORE_REJECT else (
            "低置信" if r[3] < settings.DOMAIN_SCORE_CAUTION else "放行")
        print(f"   {r[0]:<4} {r[3]:.4f}  [{verdict}]  {r[4][:30]}")

    if len(out_scope) < 10:
        print()
        print(f"⚠️ 超纲样本只有 {len(out_scope)} 条, 标定结果**统计上不可靠**。")
        print("   实测的完美分离可能只是样本太少。评测集补到 15 条以上超纲题后必须重标。")

    if args.csv:
        csv_path = Path("data/eval/reports/domain_gate_calibration.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("id,kind,answerable,vector_top1,question\n")
            for r in rows:
                fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f},\"{r[4]}\"\n")
        print(f"\n明细已写入: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
