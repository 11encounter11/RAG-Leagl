"""
LangGraph Agentic RAG 工作流

对原实现的三处关键修复:

  T-06 重排只做一次
      原实现: 每条召回路径内部都重排一次, rerank 节点又重排一次,
              单次提问最多 4 次 Cross-Encoder 推理。
      现在  : 召回统一走 retriever.retrieve_raw()(只召回不重排),
              所有候选合并去重后由 rerank 节点统一精排一次。

  T-07 不再绑定全局 retriever
      原实现: 工作流是全局单例, 创建时绑定第一个传入的 retriever。
              索引新文档后 rag_service 会换新的 retriever 实例, 但工作流仍持有
              旧实例 —— 结果就是"新上传的文档在 agentic 模式下检索不到"。
      现在  : 工作流无状态, retriever 通过 LangGraph 的 config 按请求注入。

  T-08 重试必须改变参数
      原实现: retry_count + 1 后原样重跑, query/prompt/top_k/阈值全都不变,
              大概率得到同样的低分, 只是白烧 3 倍 LLM 调用和延迟。
      现在  : 每次重试都有明确的参数变化(见 RETRY_PLAN), 且用完仍不达标就拒答,
              而不是硬生成一个没有依据的答案。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.config.settings import settings
from app.core.law_versions import detect_law_name
from app.core.llm.router import llm_router
from app.core.memory.redis_store import redis_store
from app.core.rag.retriever import (
    HybridRetriever,
    SIMILARITY_THRESHOLD_RULE,
    describe_threshold_gate,
    filter_by_similarity_threshold,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


class RAGWorkflowState(TypedDict, total=False):
    """Agentic RAG 工作流状态。total=False 让各节点只返回自己改动的键。"""

    query: str
    rewritten_query: str
    # 原问题里点名的法（规范化后的全称）；空串 = 没点名，不许在改写句上再猜
    law_name: str
    hyde_doc: str
    sub_questions: List[str]
    retrieved_docs: List[Any]
    reranked_docs: List[Any]
    context: str
    reflection_score: float
    reflection_note: str
    sources: List[dict]
    retry_count: int
    retry_params: Dict[str, Any]
    refuse: bool
    # 拒答闸门的独立判据(不依赖重排器)
    domain_score: float
    confidence_level: str
    refuse_reason: str
    # 最近一轮阈值闸门的实况(尺子名/值/是否生效/保留条数), 进 retrieval_info
    threshold_usage: Dict[str, Any]
    workflow_steps: List[str]


# 置信档位
CONFIDENCE_HIGH = "high"
CONFIDENCE_LOW = "low"
CONFIDENCE_OUT_OF_DOMAIN = "out_of_domain"


def classify_domain(score: float) -> str:
    """把领域判据分数映射成置信档位。

    两档而不是单阈值: 实测下"库内问题"和"超纲问题"的分数**区间重叠**,
    任何单阈值都必然误判一边。与其假装能一刀切, 不如把不确定显式表达出来 ——
    低置信是提示用户核对, 硬拒答才是拒绝服务。
    """
    if not settings.ENABLE_DOMAIN_GATE:
        return CONFIDENCE_HIGH
    if score < settings.DOMAIN_SCORE_REJECT:
        return CONFIDENCE_OUT_OF_DOMAIN
    if score < settings.DOMAIN_SCORE_CAUTION:
        return CONFIDENCE_LOW
    return CONFIDENCE_HIGH


# ============================================================
# 重试计划 —— 每一次重试都必须改变点什么
# ============================================================
RETRY_PLAN: List[Dict[str, Any]] = [
    # 阈值是**准入闸门**(只判这一批的 top1, 见 filter_by_similarity_threshold),
    # 不是"每条片段的及格线"。所以重试**不再动它**: 该扩大的是召回面(top_k、
    # 拆子问题), 不是降低证据标准。旧表里的 0.30 / 0.20 写于默认阈值还是 0.5 的
    # 年代, 默认改成 0.1 之后它们反而比第 0 轮更严 —— 属于参数漂移(P2-20 一并修)。
    {"round": 0, "top_k": settings.TOP_K, "threshold": settings.SIMILARITY_THRESHOLD, "split": False},
    # 第 1 轮: 扩大召回面
    {"round": 1, "top_k": 15, "threshold": settings.SIMILARITY_THRESHOLD, "split": False},
    # 第 2 轮: 复合问题拆成子问题分别召回
    {"round": 2, "top_k": 15, "threshold": settings.SIMILARITY_THRESHOLD, "split": True},
]


def params_for_round(retry_count: int) -> Dict[str, Any]:
    """取第 N 轮重试的参数(超出计划则沿用最后一档)。"""
    index = max(0, min(retry_count, len(RETRY_PLAN) - 1))
    return dict(RETRY_PLAN[index])


def _get_retriever(config: Optional[dict]) -> HybridRetriever:
    retriever = (config or {}).get("configurable", {}).get("retriever")
    if retriever is None:
        raise RuntimeError("工作流未注入 retriever(应通过 config.configurable.retriever 传入)")
    return retriever


async def _ask_llm_cached(llm, prompt: str, purpose: str) -> tuple[str, bool]:
    """问模型一次；同样的输入命中缓存就**不花这次调用**。返回 `(文本, 是否命中)`。

    为什么要缓存改写与拆分（2026-10-19 实测）：这两步的输出决定"拿哪句话去检索"，
    而模型即使 `temperature=0` 也不保证每次一样 —— 同一句问题改写三次得到三个版本
    （有的多列一个关键词、有的把"社保征收机构"丢了）。输入不稳定，agentic 路径的
    评测就没有可比性，门禁"不低于基线"无从判定。

    缓存同时解决另外两件事：同一问题重问不再重复计费；留痕能记下**当时真正用的那句**
    （原来 `rewritten_query` 连工作流的返回值都不在，排查时只能自己包一层才看得见）。

    只对"工具型调用"用这个函数。生成答案那一步不缓存 —— 那是给用户看的正文，
    重复返回同一段反而不对，而且它不影响检索。
    """
    digest = hashlib.sha1(
        f"{llm.model_name}|{purpose}|{prompt}".encode("utf-8")
    ).hexdigest()
    key = f"llm:{purpose}:{digest}"

    hit = await redis_store.cache_get(key)
    if isinstance(hit, dict) and isinstance(hit.get("text"), str) and hit["text"].strip():
        return hit["text"].strip(), True

    text = ((await llm.chat([{"role": "user", "content": prompt}])) or "").strip()
    if text:
        await redis_store.cache_set(key, {"text": text})
    return text, False


def _step(state: RAGWorkflowState, name: str) -> List[str]:
    return list(state.get("workflow_steps", [])) + [name]


def create_rag_workflow():
    """创建(无状态的)Agentic RAG 工作流。

    刻意**不接收 retriever** —— 这是 T-07 的修复点: retriever 由每次调用时的
    config 注入, 所以重新索引后不会读到旧数据。
    """

    # ============================================================
    # 节点
    # ============================================================
    async def analyze_query(state: RAGWorkflowState, config: RunnableConfig = None) -> dict:
        """查询改写 + 法名识别。

        法名只在**原问题**上判一次，写进 state 供后面所有检索变体共用：
        改写句可能把“劳动法”改没、也可能改出另一部法，让它各自判法名等于
        给跨法候选开了一条漏网通道（P2-21）。
        """
        query = state["query"]
        law_name = detect_law_name(query) or ""
        logger.info("[RAGWorkflow] 法名识别: %s", f"《{law_name}》" if law_name else "未点名")

        if not settings.ENABLE_QUERY_REWRITE:
            return {
                "rewritten_query": query,
                "law_name": law_name,
                "workflow_steps": _step(state, "query_analysis:skipped"),
            }

        llm = llm_router.get(settings.QUERY_REWRITE_MODEL)
        prompt = (
            "请把下面的用户问题改写成更适合在法律知识库中检索的查询。\n"
            "要求：\n"
            "1. 保留原问题的核心意图，不要改变事实；\n"
            "2. 补上可能的同义法律表述（例如“被开除”可补充“解除劳动合同”）；\n"
            "3. 只输出改写后的查询，不要解释、不要编号；\n"
            "4. 如果原问题已经很明确，原样返回即可。\n\n"
            f"原始问题：{query}\n\n改写后："
        )
        rewritten, from_cache = await _ask_llm_cached(llm, prompt, "query_rewrite")
        return {
            "rewritten_query": rewritten or query,
            "law_name": law_name,
            # 缓存命中要留痕: 命中时这次问答没有花改写的钱, 但**用的是历史上某次生成的句子**。
            # 排查时这两件事必须分得开。
            "workflow_steps": _step(state, "query_analysis:cache" if from_cache else "query_analysis"),
        }

    async def generate_hyde(state: RAGWorkflowState, config: RunnableConfig = None) -> dict:
        """HyDE: 生成假设性答案用于检索。默认关闭。"""
        if not settings.ENABLE_HYDE:
            return {"workflow_steps": _step(state, "hyde:skipped")}

        query = state.get("rewritten_query") or state["query"]
        llm = llm_router.get(settings.HYDE_MODEL)
        prompt = (
            "请针对下面的问题写一段简短的假设性回答（2-3 句）。\n"
            "这段回答不需要准确，目的是包含可能相关的法律术语和概念。\n\n"
            f"问题：{query}\n\n假设性回答："
        )
        hyde_doc = await llm.chat([{"role": "user", "content": prompt}])
        return {
            "hyde_doc": (hyde_doc or "").strip(),
            "workflow_steps": _step(state, "hyde_generation"),
        }

    async def hybrid_retrieve(state: RAGWorkflowState, config: RunnableConfig = None) -> dict:
        """多路召回 —— 只召回, 不重排。"""
        retriever = _get_retriever(config)
        params = state.get("retry_params") or params_for_round(0)
        query = state["query"]
        rewritten = state.get("rewritten_query") or query
        hyde_doc = state.get("hyde_doc") or ""
        steps = ["hybrid_retrieval"]

        # 法名以原问题的结论为准，改写句/子问题一律共用：
        # 空串是明确的“没点名”，不能当成“还没判过”再自动识别一次 ——
        # 否则改写句自己冒出一个法名，那一路就会绕过原问题的判断。
        if "law_name" in state:
            law_filter = state.get("law_name") or ""
        else:
            law_filter = detect_law_name(query) or ""

        queries: List[str] = [query]
        if rewritten and rewritten != query:
            queries.append(rewritten)
        if hyde_doc:
            queries.append(hyde_doc)

        # 第 2 轮重试: 把复合问题拆成子问题分别召回
        if params.get("split") and not state.get("sub_questions"):
            sub_questions = await _split_question(state["query"])
            if sub_questions:
                queries.extend(sub_questions)
                steps.append("sub_question_split")

        if law_filter:
            steps.append(f"law_filter({law_filter})")

        per_query = [
            retriever.retrieve_raw(one_query, law_name=law_filter)
            for one_query in queries
        ]

        # 跨查询合并: 用与"向量路 + BM25 路"完全相同的 RRF 算法(见 retriever 的说明)。
        #
        # 这里原先写的是"按 node_id 保留首次出现", 注释说"不同查询之间谁更相关交给
        # 重排器判断"。那个假设在两处会破:
        #   1) 重排器被关掉或加载失败(内存不足) —— 此时合并顺序 = 第一条查询的顺序,
        #      改写那一整轮的排名信息**全部作废**;
        #   2) 即使重排器开着, 它过去也是拿**原始口语问题**给候选打分(见下面 rerank),
        #      等于用效果差的那句去评判为效果好的那句召回的东西。
        # 实测(2026-10-19, L21 社保题): 《社会保险法》第六十三条在原问题下排第 11、
        # 在改写句下排第 1, 合并后仍按 11 名待着 → 反思 0.20 → 整条链拒答。
        unique_nodes = reciprocal_rank_fusion(*per_query)

        logger.info(
            "[RAGWorkflow] 召回完成: %d 路查询, RRF 合并后候选 %d 条, 参数 %s",
            len(queries),
            len(unique_nodes),
            params,
        )
        return {
            "retrieved_docs": unique_nodes,
            "workflow_steps": list(state.get("workflow_steps", [])) + steps,
        }

    async def assess_domain(state: RAGWorkflowState, config: RunnableConfig = None) -> dict:
        """拒答闸门: 用向量原始相似度判断问题是否属于本知识库领域。

        独立于重排器 —— 这是本次测试暴露的核心设计缺陷的修正:
        原实现把"要不要拒答"完全交给重排分数阈值, 而重排器是可选组件。
        实测中重排器因内存无法加载, 结果 4 条超纲题(商标/离婚/租赁/出资)
        全部返回了劳动法条文 —— 拒答准确率 0%, 而系统没有任何异常表现。

        放在召回之后、重排之前: 这一步不需要重排结果, 也不该被重排影响。
        """
        if not settings.ENABLE_DOMAIN_GATE:
            return {
                "domain_score": 1.0,
                "confidence_level": CONFIDENCE_HIGH,
                "workflow_steps": _step(state, "domain_gate:disabled"),
            }

        retriever = _get_retriever(config)
        score = retriever.vector_top_score(state["query"])
        level = classify_domain(score)
        logger.info(
            "[RAGWorkflow] 领域判据: score=%.4f -> %s (拒答线 %.3f / 低置信线 %.3f)",
            score,
            level,
            settings.DOMAIN_SCORE_REJECT,
            settings.DOMAIN_SCORE_CAUTION,
        )
        return {
            "domain_score": score,
            "confidence_level": level,
            "workflow_steps": _step(state, f"domain_gate({score:.4f}->{level})"),
        }

    async def rerank(state: RAGWorkflowState, config: RunnableConfig = None) -> dict:
        """统一精排 —— **整个流程里只在这里重排一次**。"""
        retriever = _get_retriever(config)
        params = state.get("retry_params") or params_for_round(0)
        candidates = state.get("retrieved_docs") or []
        threshold = float(params.get("threshold", settings.SIMILARITY_THRESHOLD))
        top_k = int(params.get("top_k", settings.TOP_K))

        if not candidates:
            return {
                "reranked_docs": [],
                "context": "",
                "threshold_usage": {},
                "workflow_steps": _step(state, "rerank:no_candidates"),
            }

        # 打分的问句: 用**改写后的**那一句(没有改写时退回原句)。
        # 原先固定用 state["query"] —— 可候选池是按改写句召回来的, 拿原口语句去给它们
        # 打分, 等于"换一个问题来评判上一句找到的东西", 改写捞回来的高分条文会被
        # 系统性压低。这与上面合并的修正是同一个病的两半。
        score_against = state.get("rewritten_query") or state["query"]
        reranked = retriever.rerank(
            score_against, candidates, top_n=max(top_k, settings.RERANK_TOP_N)
        )
        # 阈值规则与直连检索共用同一处定义(见 filter_by_similarity_threshold)。
        # 之前这里自己写了一遍筛选, 漏掉"重排未生效"的防护 —— 重排关闭时分数是
        # RRF 名次分(第一名也才 0.03), 被 0.5/0.3/0.2 三轮阈值全部误杀, 表现是
        # **所有正常问题都被静默拒答**, 而直连检索那条路因为有防护一切正常,
        # 于是离线评测(跑的是直连)完全测不到。2026-10-17 实测确认。
        scores_are_relevance = retriever.rerank_effective
        kept = filter_by_similarity_threshold(
            reranked, threshold, scores_are_relevance=scores_are_relevance
        )

        final = kept[:top_k]
        usage = describe_threshold_gate(
            threshold=threshold,
            scores_are_relevance=scores_are_relevance,
            applied=bool(scores_are_relevance and threshold and threshold > 0),
            nodes=reranked,
            kept=len(kept),
            returned=len(final),
        )
        return {
            "reranked_docs": final,
            "context": retriever.get_context_text(final) if final else "",
            "threshold_usage": usage,
            "workflow_steps": _step(
                state,
                f"rerank(threshold={threshold},top_k={top_k},rule={SIMILARITY_THRESHOLD_RULE})",
            ),
        }

    async def reflect(state: RAGWorkflowState, config: RunnableConfig = None) -> dict:
        """反思: 评估检索结果是否足以回答问题。"""
        reranked_docs = state.get("reranked_docs") or []
        if not reranked_docs:
            return {
                "reflection_score": 0.0,
                "reflection_note": "没有召回任何片段",
                "workflow_steps": _step(state, "reflection:empty"),
            }

        llm = llm_router.get(settings.REFLECTION_MODEL)
        docs_text = "\n---\n".join(
            f"[片段{i + 1}]（相关性 {node.score:.3f}）\n{node.node.text[:300]}"
            for i, node in enumerate(reranked_docs[:3])
        )
        prompt = (
            "请评估下列检索片段是否足以回答用户问题。\n\n"
            f"用户问题：{state['query']}\n\n检索片段：\n{docs_text}\n\n"
            "只输出 JSON，不要任何其他内容：\n"
            '{"analysis": "一句话分析", "confidence": 0.0}\n'
            "confidence 取 0~1：1 表示片段已足够回答问题，0 表示完全不够。"
        )
        response = await llm.chat([{"role": "user", "content": prompt}])
        score, note = _parse_reflection(response)

        return {
            "reflection_score": score,
            "reflection_note": note,
            "workflow_steps": _step(state, f"reflection({score:.2f})"),
        }

    def decide(state: RAGWorkflowState) -> str:
        """条件边: 够好就生成, 不够好且还有额度就重试, 用完了就拒答。"""
        # 领域判决优先于一切:
        # 问题不在本知识库范围内时, 重试再多次也召不回"不存在的依据",
        # 只会白烧 LLM 调用。所以先拒答, 不进入重试循环。
        if state.get("confidence_level") == CONFIDENCE_OUT_OF_DOMAIN:
            return "refuse"

        score = state.get("reflection_score", 0.0)
        retry_count = state.get("retry_count", 0)
        if score >= settings.REFLECTION_THRESHOLD:
            return "generate"
        if retry_count < settings.MAX_RETRIES:
            return "retry"
        return "refuse"

    async def retry_handler(state: RAGWorkflowState, config: RunnableConfig = None) -> dict:
        """重试: 推进到下一档参数, 清空上一轮结果。"""
        retry_count = state.get("retry_count", 0) + 1
        new_params = params_for_round(retry_count)
        logger.info("[RAGWorkflow] 第 %d 次重试, 参数 -> %s", retry_count, new_params)
        return {
            "retry_count": retry_count,
            "retry_params": new_params,
            "retrieved_docs": [],
            "reranked_docs": [],
            "context": "",
            "sub_questions": [],
            "workflow_steps": _step(state, f"retry#{retry_count}({new_params})"),
        }

    async def refuse(state: RAGWorkflowState, config: RunnableConfig = None) -> dict:
        """拒答: 依据不足或问题超出知识库范围时明确告知, 而不是编造。"""
        out_of_domain = state.get("confidence_level") == CONFIDENCE_OUT_OF_DOMAIN
        reason = (
            f"问题超出知识库范围(领域判据 {state.get('domain_score', 0.0):.4f} "
            f"< 拒答线 {settings.DOMAIN_SCORE_REJECT})"
            if out_of_domain
            else f"检索依据不足(重试 {state.get('retry_count', 0)} 次, "
            f"反思得分 {state.get('reflection_score', 0.0):.2f})"
        )
        logger.info("[RAGWorkflow] 判定拒答: %s", reason)
        return {
            "refuse": True,
            "refuse_reason": reason,
            "sources": [],
            "context": "",
            "reranked_docs": [],
            "workflow_steps": _step(state, f"refuse({reason})"),
        }

    async def generate(state: RAGWorkflowState, config: RunnableConfig = None) -> dict:
        """生成前的收尾: 整理引用来源。

        真正的 LLM 生成在 chat_service 里做(为了支持流式), 这里只准备状态。
        """
        retriever = _get_retriever(config)
        reranked_docs = state.get("reranked_docs") or []
        sources = retriever.get_sources_info(reranked_docs) if reranked_docs else []
        return {
            "refuse": False,
            "refuse_reason": "",
            "sources": sources,
            "context": state.get("context")
            or (retriever.get_context_text(reranked_docs) if reranked_docs else ""),
            "workflow_steps": _step(state, "generation_ready"),
        }

    # ============================================================
    # 组装图
    # ============================================================
    workflow = StateGraph(RAGWorkflowState)

    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("generate_hyde", generate_hyde)
    workflow.add_node("hybrid_retrieve", hybrid_retrieve)
    workflow.add_node("assess_domain", assess_domain)
    workflow.add_node("rerank", rerank)
    workflow.add_node("reflect", reflect)
    workflow.add_node("retry", retry_handler)
    workflow.add_node("refuse", refuse)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "generate_hyde")
    workflow.add_edge("generate_hyde", "hybrid_retrieve")
    # 召回后先过独立拒答闸门, 再重排。
    # 顺序上刻意如此: 闸门只看向量原始相似度, 不需要也不应该等重排结果 ——
    # 否则重排器一挂, 闸门就跟着失效。
    workflow.add_edge("hybrid_retrieve", "assess_domain")
    workflow.add_edge("assess_domain", "rerank")
    workflow.add_edge("rerank", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        decide,
        {"generate": "generate", "retry": "retry", "refuse": "refuse"},
    )
    # 重试直接回到 hybrid_retrieve, **不回到 analyze_query**。
    # 改写用的 prompt 没变, 重跑一次只会得到同样的结果, 纯属浪费一次 LLM 调用。
    # 真正需要改变的是检索参数(见 RETRY_PLAN)。
    workflow.add_edge("retry", "hybrid_retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("refuse", END)

    return workflow.compile()


# ============================================================
# 辅助
# ============================================================
async def _split_question(query: str) -> List[str]:
    """把复合问题拆成子问题(重试第 2 轮使用)。"""
    try:
        llm = llm_router.get(settings.QUERY_REWRITE_MODEL)
        prompt = (
            "下面这个问题可能同时包含多个法律问题。请把它拆成最多 3 个独立子问题，"
            "每行一个，不要编号，不要解释。如果它本身只包含一个问题，就原样输出一行。\n\n"
            f"问题：{query}"
        )
        response, _ = await _ask_llm_cached(llm, prompt, "question_split")
        lines = [line.strip(" -•\t") for line in (response or "").splitlines()]
        return [line for line in lines if line and line != query][:3]
    except Exception as exc:  # noqa: BLE001 - 拆解失败不该让整个请求失败
        logger.warning("[RAGWorkflow] 子问题拆解失败, 跳过: %s", exc)
        return []


def _parse_reflection(response: str) -> tuple[float, str]:
    """从反思节点输出里解析置信度。

    解析不出来时保守地返回 0.0(而不是 0.5)—— 反思节点的作用是"决定能不能答",
    解析失败就当成没把握, 交给重试与拒答逻辑处理, 而不是乐观地放行。
    """
    text = (response or "").strip()
    if not text:
        return 0.0, "反思节点无输出"

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()

    try:
        payload = json.loads(text)
        score = float(payload.get("confidence", 0.0))
        return max(0.0, min(1.0, score)), str(payload.get("analysis", ""))[:200]
    except Exception:  # noqa: BLE001
        match = re.search(r"confidence\"?\s*[:：]\s*([0-9]*\.?[0-9]+)", text)
        if match:
            return max(0.0, min(1.0, float(match.group(1)))), text[:200]
        return 0.0, f"无法解析反思输出: {text[:120]}"


# ============================================================
# 工作流缓存 —— 按开关取值缓存, 改了开关会重建
# ============================================================
_WORKFLOW_CACHE: Dict[tuple, Any] = {}


def _flag_signature() -> tuple:
    return (
        settings.ENABLE_AGENTIC_RAG,
        settings.ENABLE_QUERY_REWRITE,
        settings.ENABLE_HYDE,
        settings.ENABLE_RERANK,
        settings.MAX_RETRIES,
        settings.REFLECTION_THRESHOLD,
        settings.ENABLE_DOMAIN_GATE,
        settings.DOMAIN_SCORE_REJECT,
        settings.DOMAIN_SCORE_CAUTION,
    )


def get_rag_workflow():
    """获取工作流实例。

    **签名里不接收 retriever**(T-07 的修复点): 工作流本身无状态, retriever
    由每次调用的 config 注入, 所以重新索引后不会读到旧数据。
    缓存键包含开关取值, 改了开关会自动重建。
    """
    key = _flag_signature()
    if key not in _WORKFLOW_CACHE:
        _WORKFLOW_CACHE.clear()
        _WORKFLOW_CACHE[key] = create_rag_workflow()
        logger.info("[RAGWorkflow] 工作流已创建, 开关配置 = %s", key)
    return _WORKFLOW_CACHE[key]
