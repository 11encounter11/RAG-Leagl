"""
重排器探针 —— 回答三个"看代码看不出来"的问题

为什么需要这个探针:
    重排器加载失败时的表现是 Rust 层 panic 或进程被杀, 不是 Python 异常栈。
    这类失败在代码审查、单元测试里都看不见 —— 只有真的加载一次才知道。
    同样, "重排分数是 logit 还是已经过 Sigmoid" 决定了判定阈值的含义,
    配置项 RERANK_SCORE_IS_LOGIT 猜错会让阈值语义整体错位。

测什么:
    1. 加载能不能成功, 耗时多少, 吃多少内存(峰值)
    2. 加载成功后真的排一次序 —— "能加载"不等于"能工作"
    3. 分数落在什么区间 —— 用来确定 RERANK_SCORE_IS_LOGIT 该设什么

用法:
    python scripts/probe_reranker.py
    python scripts/probe_reranker.py --skip-load   # 只看内存与环境, 不加载
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402


def mem_report(tag: str) -> dict:
    import psutil

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    snap = {
        "tag": tag,
        "avail_gb": round(vm.available / 1024**3, 2),
        "used_pct": vm.percent,
        "swap_free_gb": round(sw.free / 1024**3, 2),
    }
    print(
        f"  [{tag}] 可用 {snap['avail_gb']:.2f} GB | 占用 {snap['used_pct']}% | "
        f"交换区可用 {snap['swap_free_gb']:.2f} GB"
    )
    return snap


def main() -> int:
    parser = argparse.ArgumentParser(description="重排器探针")
    parser.add_argument("--skip-load", action="store_true", help="只测环境, 不加载模型")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="临时覆盖 RERANK_TORCH_DTYPE, 用于实测不同精度的内存占用差异(不改动 .env)",
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "onnx", "openvino"],
        default=None,
        help="临时覆盖 RERANK_BACKEND",
    )
    args = parser.parse_args()

    if args.dtype:
        settings.RERANK_TORCH_DTYPE = args.dtype
    if args.backend:
        settings.RERANK_BACKEND = args.backend

    print("=" * 70)
    print("重排器探针")
    print("=" * 70)
    print(f"  RERANK_MODEL            = {settings.RERANK_MODEL}")
    print(f"  RERANK_CANDIDATE_LIMIT  = {settings.RERANK_CANDIDATE_LIMIT}")
    print(f"  RERANK_SCORE_IS_LOGIT   = {settings.RERANK_SCORE_IS_LOGIT}")
    print(f"  RERANK_TORCH_DTYPE      = {settings.RERANK_TORCH_DTYPE}")
    print(f"  RERANK_BACKEND          = {settings.RERANK_BACKEND}")
    print(f"  ENABLE_RERANK           = {settings.ENABLE_RERANK}")
    print()
    print(
        "  提示: 若模型已在本地缓存, 建议加环境变量 HF_HUB_OFFLINE=1 运行本脚本。\n"
        "        否则 sentence-transformers 每次都会先发 HEAD 请求去 huggingface.co\n"
        "        核对文件是否有更新, 网络不通时会重试 5 次(每次退避翻倍),\n"
        "        表现为'卡住几十秒后才有输出', 容易被误判成加载本身很慢。"
    )
    print()

    # 模型是否已经在本地 —— 这条能区分"下载失败"和"内存不足",
    # 两者表现都是"重排器不可用", 但修法完全不同。
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_dir = cache / f"models--{settings.RERANK_MODEL.replace('/', '--')}"
    if model_dir.exists():
        size_mb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1024**2
        print(f"  ✅ 模型已在本地: {model_dir}  ({size_mb:.0f} MB)")
        weights = [f.name for f in model_dir.rglob("*.safetensors")] + [
            f.name for f in model_dir.rglob("*.bin")
        ]
        print(f"     权重文件: {weights}")
    else:
        print(f"  ⚠️ 模型不在本地, 首次加载需要联网下载: {model_dir}")
    print()

    print("内存基线:")
    baseline = mem_report("加载前")
    print()

    if args.skip_load:
        print("(--skip-load 已指定, 不加载模型)")
        return 0

    # ------------------------------------------------------------------
    # 真实加载路径: 走应用自己的代码, 不用旁路
    # ------------------------------------------------------------------
    print("步骤 1: 加载(走应用真实路径, 非旁路)")
    print("-" * 70)
    from app.core.rag.retriever import HybridRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    class _NullRetriever:
        def retrieve(self, query):  # 本探针只测重排器, 不碰向量检索
            return []

    class _NullIndex:
        def as_retriever(self, similarity_top_k=None):
            return _NullRetriever()

    t0 = time.time()
    retriever = HybridRetriever(
        index=_NullIndex(),
        nodes=[],
        use_reranker=True,
    )
    ok = False
    try:
        ok = retriever.preload_reranker()
    except BaseException as exc:  # noqa: BLE001
        # 特意捕获 BaseException: 上次的失败是 Rust 层 panic,
        # 它未必继承自 Exception, 用 except Exception 会漏掉。
        print(f"  ❌ 加载抛出 {type(exc).__name__}: {exc}")
    elapsed = time.time() - t0
    print(f"  加载结果 = {ok}, 耗时 {elapsed:.1f}s")
    after_load = mem_report("加载后")
    peak_delta = baseline["avail_gb"] - after_load["avail_gb"]
    print(f"  净增内存占用 = {peak_delta:.2f} GB")
    print()

    if not ok:
        print("=" * 70)
        print("❌ 重排器不可用 —— 该能力在当前环境下无法验证")
        print("=" * 70)
        print("  ⚠️ 注意: 系统不会因此报错, 只会**静默跳过重排**。")
        print("     如果拒答闸门依赖重排分数, 就会连带失效 —— 这正是 N-10 的成因。")
        print("     当前拒答闸门已改为独立判据(向量原始相似度), 不再受此影响。")
        return 2

    # ------------------------------------------------------------------
    # 步骤 2: 真的排一次序 —— "能加载"不等于"能工作"
    # ------------------------------------------------------------------
    print("步骤 2: 实际排序(能加载 ≠ 能工作)")
    print("-" * 70)
    query = "试用期最长可以多久"
    nodes = [
        NodeWithScore(
            node=TextNode(text="劳动合同期限三个月以上不满一年的，试用期不得超过一个月。"),
            score=0.9,
        ),
        NodeWithScore(
            node=TextNode(text="三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月。"),
            score=0.4,
        ),
        NodeWithScore(
            node=TextNode(text="劳动合同期限一年以上不满三年的，试用期不得超过二个月。"),
            score=0.7,
        ),
        NodeWithScore(node=TextNode(text="今天天气不错，适合出去散步。"), score=0.8),
    ]
    try:
        ranked = retriever.rerank(query, nodes)
    except BaseException as exc:  # noqa: BLE001
        print(f"  ❌ 排序抛出 {type(exc).__name__}: {exc}")
        return 3

    print(f"  查询: {query}")
    for i, n in enumerate(ranked, 1):
        txt = n.node.text.replace("\n", " ")[:36]
        print(f"    {i}. score={n.score:>8.4f}  {txt}")

    scores = [n.score for n in ranked]
    # 期望: "六个月"那条最前(它才是"最长"), "二个月"次之, "一个月"再次, 天气最后。
    # 提示: 这里必须放**真正的最优答案**, 否则测的是我的语文水平不是重排器。
    order_ok = ranked[0].node.text.startswith("三年以上固定期限")
    irrelevant_last = "天气" in ranked[-1].node.text
    print()
    print(f"  最优答案排第一: {'✅ 是' if order_ok else '❌ 否'}")
    print(f"    (期望「不得超过六个月」那条排第一 —— 它才是「最长」的答案)")
    print(f"  无关内容排最后: {'✅ 是' if irrelevant_last else '❌ 否'}")
    print(f"  输入顺序已被打乱: 原始 score 0.9/0.4/0.7/0.8 → 重排后见上, "
          f"{'✅ 顺序确实变了' if not order_ok or ranked[0].score != 0.9 else '⚠️ 未见变化'}")

    # ------------------------------------------------------------------
    # 步骤 3: 分数语义 —— 决定配置项怎么写
    # ------------------------------------------------------------------
    print()
    print("步骤 3: 分数语义(决定 RERANK_SCORE_IS_LOGIT 该设什么)")
    print("-" * 70)
    lo, hi = min(scores), max(scores)
    in_unit = 0.0 <= lo and hi <= 1.0
    print(f"  实测分数区间: [{lo:.4f}, {hi:.4f}]")
    if in_unit:
        print("  → 全部落在 [0,1], 判断为**已过 Sigmoid**")
        print(f"  → 配置应为 RERANK_SCORE_IS_LOGIT = False (当前值: {settings.RERANK_SCORE_IS_LOGIT})")
        verdict = "False"
    else:
        print("  → 存在区间外的值, 判断为**未过 Sigmoid 的原始 logit**")
        print(f"  → 配置应为 RERANK_SCORE_IS_LOGIT = True (当前值: {settings.RERANK_SCORE_IS_LOGIT})")
        verdict = "True"

    if str(settings.RERANK_SCORE_IS_LOGIT) != verdict:
        print()
        print(f"  ⚠️ 配置与实际不符! 应将 RERANK_SCORE_IS_LOGIT 改为 {verdict}")
        print("     语义搞错会让归一化在 0 点不连续, 阈值的含义整体错位。")
    else:
        print("  ✅ 配置与实际一致")

    print()
    print("=" * 70)
    print("✅ 重排器可用 —— 该项能力可以进入验收")
    print(f"   加载耗时 {elapsed:.1f}s, 净增占用 {peak_delta:.2f} GB")
    print("=" * 70)

    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
