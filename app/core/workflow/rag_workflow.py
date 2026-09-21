"""
LangGraph Agentic RAG 工作流
实现：查询改写 → HyDE → 混合检索 → 重排序 → 反思 → 生成
具备自主决策能力：自动评估检索质量，不够好则自动重试
"""
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from app.core.llm.router import llm_router
from app.core.rag.retriever import HybridRetriever
from app.config.settings import settings


class RAGWorkflowState(TypedDict):
    """Agentic RAG 工作流状态"""
    query: str                    # 原始问题
    rewritten_query: str          # 改写后的查询
    hyde_doc: str                 # HyDE 假设文档
    retrieved_docs: list          # 召回的文档节点
    reranked_docs: list           # 重排序后的文档
    context: str                  # 上下文文本
    reflection_score: float       # 反思置信度
    answer: str                   # 生成的答案
    sources: list                 # 来源引用
    retry_count: int              # 重试次数
    workflow_steps: list          # 工作流步骤记录


def create_rag_workflow(retriever: HybridRetriever):
    """
    创建 Agentic RAG 工作流
    Args:
        retriever: 混合检索器实例
    Returns:
        编译后的 LangGraph 工作流
    """

    # ===== 节点函数 =====

    async def analyze_query(state: RAGWorkflowState) -> dict:
        """
        查询分析与改写节点
        使用 LLM 优化用户查询，提升检索效果
        """
        query = state["query"]
        llm = llm_router.get(settings.QUERY_REWRITE_MODEL)

        prompt = f"""请优化以下用户问题，使其更适合在知识库中检索相关信息。
要求：
1. 保留原问题的核心意图
2. 补充可能的同义词或相关概念
3. 使问题更加具体和明确
4. 直接输出优化后的问题，不要解释

原始问题：{query}

优化后的问题："""

        messages = [{"role": "user", "content": prompt}]
        rewritten = await llm.chat(messages)

        return {
            "rewritten_query": rewritten.strip(),
            "workflow_steps": state.get("workflow_steps", []) + ["query_analysis"]
        }

    async def generate_hyde(state: RAGWorkflowState) -> dict:
        """
        HyDE (Hypothetical Document Embedding) 节点
        生成假设性答案文档，用于增强检索
        """
        query = state.get("rewritten_query") or state["query"]
        llm = llm_router.get(settings.HYDE_MODEL)

        prompt = f"""请针对以下问题生成一个简短的假设性回答（2-3句话）。
这个回答不需要完全准确，但要包含可能相关的关键词和概念。

问题：{query}

假设性回答："""

        messages = [{"role": "user", "content": prompt}]
        hyde_doc = await llm.chat(messages)

        return {
            "hyde_doc": hyde_doc.strip(),
            "workflow_steps": state.get("workflow_steps", []) + ["hyde_generation"]
        }

    def hybrid_retrieve(state: RAGWorkflowState) -> dict:
        """
        混合检索节点
        使用原始查询、改写查询、HyDE文档进行多路召回
        """
        query = state["query"]
        rewritten_query = state.get("rewritten_query", query)
        hyde_doc = state.get("hyde_doc", "")

        # 多路检索
        all_nodes = []
        
        # 路径1: 原始查询
        nodes1 = retriever.retrieve(query, top_k=settings.TOP_K)
        all_nodes.extend(nodes1)

        # 路径2: 改写查询
        if rewritten_query and rewritten_query != query:
            nodes2 = retriever.retrieve(rewritten_query, top_k=settings.TOP_K)
            all_nodes.extend(nodes2)

        # 路径3: HyDE 文档
        if hyde_doc:
            nodes3 = retriever.retrieve(hyde_doc, top_k=settings.TOP_K)
            all_nodes.extend(nodes3)

        # 去重（按 node_id）
        seen_ids = set()
        unique_nodes = []
        for node in all_nodes:
            node_id = node.node.node_id
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_nodes.append(node)

        return {
            "retrieved_docs": unique_nodes,
            "workflow_steps": state.get("workflow_steps", []) + ["hybrid_retrieval"]
        }

    def rerank(state: RAGWorkflowState) -> dict:
        """
        重排序节点
        使用 Cross-Encoder 对候选文档进行精排
        """
        query = state["query"]
        retrieved_docs = state["retrieved_docs"]

        if not retrieved_docs:
            return {
                "reranked_docs": [],
                "context": "",
                "workflow_steps": state.get("workflow_steps", []) + ["reranking"]
            }

        # 重排序
        reranked = retriever._rerank(
            query, 
            retrieved_docs, 
            top_n=settings.RERANK_TOP_N
        )

        # 生成上下文
        context = retriever.get_context_text(reranked)

        return {
            "reranked_docs": reranked,
            "context": context,
            "workflow_steps": state.get("workflow_steps", []) + ["reranking"]
        }

    async def reflect(state: RAGWorkflowState) -> dict:
        """
        反思节点
        评估检索结果质量，决定是否重试
        """
        query = state["query"]
        reranked_docs = state["reranked_docs"]

        if not reranked_docs:
            return {
                "reflection_score": 0.0,
                "workflow_steps": state.get("workflow_steps", []) + ["reflection"]
            }

        llm = llm_router.get(settings.QUERY_REWRITE_MODEL)

        # 构建评估文本
        docs_text = "\n---\n".join([
            f"[文档{i+1}] (相关度: {node.score:.3f})\n{node.node.text[:200]}"
            for i, node in enumerate(reranked_docs[:3])
        ])

        prompt = f"""请评估以下检索结果与用户问题的相关性。

用户问题：{query}

检索结果：
{docs_text}

请分析：
1. 检索结果是否包含回答问题所需的信息？
2. 文档内容与问题的匹配程度如何？
3. 给出一个 0-1 之间的置信度分数（1表示非常相关，0表示完全不相关）

请以JSON格式回答：
{{"analysis": "简要分析", "confidence": 0.XX}}"""

        messages = [{"role": "user", "content": prompt}]
        response = await llm.chat(messages)

        # 解析置信度分数
        import json
        import re
        
        try:
            result = json.loads(response)
            score = float(result.get("confidence", 0.5))
        except:
            # 尝试从文本中提取分数
            match = re.search(r'confidence["\s:]+(\d+\.?\d*)', response)
            score = float(match.group(1)) if match else 0.5

        return {
            "reflection_score": score,
            "workflow_steps": state.get("workflow_steps", []) + ["reflection"]
        }

    def should_retry(state: RAGWorkflowState) -> str:
        """
        条件边：判断是否重试
        """
        score = state.get("reflection_score", 0.0)
        retry_count = state.get("retry_count", 0)

        # 置信度足够高，进入生成
        if score >= settings.REFLECTION_THRESHOLD:
            return "generate"
        
        # 还有重试机会，回到查询分析
        if retry_count < settings.MAX_RETRIES:
            return "retry"
        
        # 已达最大重试次数，强制生成
        return "generate"

    async def generate(state: RAGWorkflowState) -> dict:
        """
        生成节点（工作流终点）
        注意：实际的 LLM 生成在 chat_service 中完成（支持流式）
        这里只准备工作流状态
        """
        sources = retriever.get_sources_info(state["reranked_docs"])
        
        return {
            "sources": sources,
            "workflow_steps": state.get("workflow_steps", []) + ["generation_ready"]
        }

    async def retry_handler(state: RAGWorkflowState) -> dict:
        """
        重试处理器
        增加重试计数，清空之前的检索结果
        """
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "retrieved_docs": [],
            "reranked_docs": [],
            "context": "",
            "workflow_steps": state.get("workflow_steps", []) + ["retry"]
        }

    # ===== 构建状态图 =====

    workflow = StateGraph(RAGWorkflowState)

    # 添加节点
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("generate_hyde", generate_hyde)
    workflow.add_node("hybrid_retrieve", hybrid_retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("reflect", reflect)
    workflow.add_node("generate", generate)
    workflow.add_node("retry", retry_handler)

    # 添加边
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "generate_hyde")
    workflow.add_edge("generate_hyde", "hybrid_retrieve")
    workflow.add_edge("hybrid_retrieve", "rerank")
    workflow.add_edge("rerank", "reflect")

    # 条件边：反思后决定是否重试
    workflow.add_conditional_edges(
        "reflect",
        should_retry,
        {
            "generate": "generate",
            "retry": "retry",
        }
    )

    # 重试后回到查询分析
    workflow.add_edge("retry", "analyze_query")

    # 生成后结束
    workflow.add_edge("generate", END)

    # 编译工作流
    return workflow.compile()


# 全局工作流实例（延迟创建）
_workflow_instance = None

def get_rag_workflow(retriever: HybridRetriever):
    """获取全局工作流实例"""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = create_rag_workflow(retriever)
    return _workflow_instance
