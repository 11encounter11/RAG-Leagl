"""
混合检索器 - 向量检索 + BM25关键词检索 + RRF融合 + Cross-Encoder重排序

本模块的三条设计约束(都是对原实现的修复):
    1. **召回与重排分离** —— retrieve_raw() 只做多路召回, 重排由 rerank() 显式调用。
       原因: 工作流要对原始 query / 改写 query / HyDE 文档分别召回,
       原实现里每条路内部都重排一次, 之后又重排一次, 单次提问最多跑 4 次
       Cross-Encoder, 首 token 延迟成倍增加。重排应在所有召回合并后统一做一次。(T-06)
    2. **分数语义必须明确** —— Cross-Encoder 的输出量纲与向量余弦相似度不同。
       阈值只能作用在归一化之后的分上, 否则那个阈值旋钮没有物理意义。(T-02)
    3. **BM25 语料可重建** —— 语料不再只活在进程内存里(重启即失效),
       可由 Milvus 中的正文重建。(T-10)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode
from rank_bm25 import BM25L

from app.config.settings import settings

logger = logging.getLogger(__name__)


class _CrossEncoderReranker:
    """对 `sentence_transformers.CrossEncoder` 的轻量封装。

    为什么不直接用 llama-index 的 `SentenceTransformerRerank`:
        它的构造函数不暴露 `model_kwargs` / `backend` 两个参数, 而这两个参数
        正是控制重排器**常驻内存占用**的唯一旋钮(加载精度、推理后端)。
        N-9 里 bge-reranker-base 以 float32 加载约 1GB 常驻内存, 曾把开发机
        逼到 OOM; 想把权重精度降到 float16/bfloat16 来砍半内存, 就必须绕过
        llama-index 这层封装, 直接构造 CrossEncoder。
    对外仍保持 `postprocess_nodes(nodes, query_str=...)` 这一签名,
    与 `SentenceTransformerRerank` 一致, 单元测试里的 `_FakeReranker` 无需改动。
    """

    def __init__(
        self,
        model: str,
        top_n: int,
        *,
        backend: str = "torch",
        torch_dtype: str = "float32",
        keep_retrieval_score: bool = True,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model = model
        self.top_n = top_n
        self.keep_retrieval_score = keep_retrieval_score

        model_kwargs: dict[str, Any] = {}
        if backend == "torch" and torch_dtype != "float32":
            import torch

            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
            if torch_dtype not in dtype_map:
                raise ValueError(
                    f"不支持的 RERANK_TORCH_DTYPE={torch_dtype!r}, 可选: float32/float16/bfloat16"
                )
            model_kwargs["torch_dtype"] = dtype_map[torch_dtype]

        self._model = CrossEncoder(
            model,
            backend=backend,
            model_kwargs=model_kwargs or None,
            trust_remote_code=True,
        )
        self.torch_dtype = torch_dtype
        self.backend = backend

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        if query_str is None:
            raise ValueError("缺少 query_str")
        if not nodes:
            return []

        pairs = [
            (query_str, node.node.get_content(metadata_mode=MetadataMode.EMBED))
            for node in nodes
        ]
        scores = self._model.predict(pairs)
        assert len(scores) == len(nodes)

        for node, score in zip(nodes, scores):
            if self.keep_retrieval_score:
                node.node.metadata["retrieval_score"] = node.score
            node.score = float(score)

        return sorted(nodes, key=lambda x: -(x.score or 0.0))[: self.top_n]


# ============================================================
# 分数归一化
# ============================================================
def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 700.0)))
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


def normalize_rerank_score(raw: float | None, *, treat_as_logit: bool | None = None) -> float:
    """把 Cross-Encoder 的输出归一化成 [0, 1] 的置信度。

    为什么不能靠"逐值判断是不是 logits":
        曾想过"值落在 [0,1] 就当成概率, 否则套 sigmoid"。但这样在 0 处不连续 ——
        sigmoid(0) = 0.5, 而返回值会是 0.0, 阈值立刻失去单调意义。
    所以改成**整批一次性确定语义**:
        · RERANK_SCORE_IS_LOGIT = False(默认): 认为上游已套过 Sigmoid(bge-reranker
          的 config 里带 sbert_ce_default_activation_function), 只做边界钳制;
        · RERANK_SCORE_IS_LOGIT = True : 上游返回原始 logits, 施加 sigmoid。
    两种模式下都是单调的, 阈值都有明确物理含义。

    >>> round(normalize_rerank_score(-1.0, treat_as_logit=True), 4)
    0.2689
    >>> normalize_rerank_score(0.8)
    0.8
    >>> normalize_rerank_score(3.0)   # 默认按概率处理, 越界只钳制
    1.0
    """
    if raw is None:
        return 0.0
    value = float(raw)
    if treat_as_logit is None:
        treat_as_logit = settings.RERANK_SCORE_IS_LOGIT
    if treat_as_logit:
        return _sigmoid(value)
    return min(1.0, max(0.0, value))


# ============================================================
# 中文分词
# ============================================================
def tokenize_zh(text: str) -> List[str]:
    """中文分词, 供 BM25 使用。

    原实现用字符级单字: 单字粒度会让 IDF 几乎失效(常用字到处都是),
    长文档天然占优, BM25 这一路的精度很差。改用 jieba 的搜索引擎模式;
    jieba 不可用时退回字 bigram —— bigram 已能捕捉相当一部分词组信息。
    """
    text = (text or "").strip()
    if not text:
        return []
    try:
        import jieba

        return [token for token in (t.strip() for t in jieba.lcut_for_search(text)) if token]
    except Exception:  # pragma: no cover - 仅在 jieba 缺失时走到
        cleaned = text.replace(" ", "")
        return [cleaned[i : i + 2] for i in range(max(len(cleaned) - 1, 1))]


class HybridRetriever:
    """混合检索器: 向量语义召回 + BM25 关键词召回 -> RRF 融合 -> Cross-Encoder 精排。"""

    def __init__(
        self,
        index: VectorStoreIndex,
        nodes: Optional[List[TextNode]] = None,
        use_reranker: bool = True,
    ) -> None:
        self._index = index
        self._nodes: List[TextNode] = list(nodes or [])
        self._use_reranker = use_reranker
        self._reranker: Any = None
        self._reranker_failed = False
        self._score_semantics_checked = False
        # 上一次 rerank() 是否**真的**执行了重排。
        # 为 False 时返回的分数仍是 RRF 原值(0.01~0.03 量级), 不是归一化相似度,
        # 因而绝不能施加相似度阈值 —— 否则整路召回被静默误杀(见 retrieve 的说明)。
        self._last_rerank_ok = False
        self._bm25: Optional[BM25L] = None

        self._vector_top_k = max(settings.BM25_TOP_K * 2, settings.TOP_K)
        # 这个 retriever **不带**时效过滤，供拒答闸门使用（见 vector_top_score 的说明）。
        self._vector_retriever = self._index.as_retriever(similarity_top_k=self._vector_top_k)
        if self._nodes:
            self.build_bm25_index(self._nodes)

    # ------------------------------------------------------------
    # BM25 语料
    # ------------------------------------------------------------
    def build_bm25_index(self, nodes: Sequence[TextNode]) -> None:
        """(重)建 BM25 索引。

        用 BM25L 而不是 BM25Okapi: Okapi 的 idf 公式是
            log((N - df + 0.5) / (df + 0.5))
        当某个词出现在超过半数文档里时 idf 为负。小知识库(尤其单一主题的法条库,
        "劳动者""用人单位"几乎篇篇都有)极易触发 —— 结果 get_scores 返回负数,
        下游按 "score > 0" 过滤就会把整路召回清空, 而且**不报错**, 只是关键词
        检索永远返回空。BM25L 的 idf 恒非负, 不存在这个问题。
        """
        self._nodes = list(nodes)
        if not self._nodes:
            self._bm25 = None
            logger.warning("[HybridRetriever] BM25 语料为空, 关键词召回这一路不可用")
            return
        tokenized = [tokenize_zh(node.text or "") for node in self._nodes]
        self._bm25 = BM25L(tokenized)
        logger.info("[HybridRetriever] BM25 索引构建完成, 语料 %d 条", len(self._nodes))

    def set_bm25_corpus_from_records(self, records: Iterable[dict]) -> int:
        """从 Milvus 记录重建 BM25 语料。

        records 形如 [{"id":..., "text":..., "metadata": {...}}, ...]。
        这样重启后 BM25 不再失效(原实现依赖进程内存, 重启即为空, 混合检索
        会静默退化成纯向量检索 —— 这是最难发现的一类问题: 不报错, 只是变差)。
        """
        nodes: List[TextNode] = []
        for record in records:
            text = record.get("text") or ""
            if not text:
                continue
            nodes.append(
                TextNode(
                    id_=str(record.get("id") or "") or None,
                    text=text,
                    metadata=record.get("metadata") or {},
                )
            )
        self.build_bm25_index(nodes)
        return len(nodes)

    @property
    def bm25_ready(self) -> bool:
        return self._bm25 is not None and len(self._nodes) > 0

    @property
    def bm25_size(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------
    def vector_top_score(self, query: str) -> float:
        """只看向量那一路的最高原始相似度, 用作"是否属于本知识库领域"的判据。

        刻意不经过 RRF 融合、不经过重排器:
          - RRF 分数是排名倒数之和(量级 0.01), 与"像不像本领域"无关;
          - 重排器是可选组件, 而拒答是安全底线, 不能依赖它。
        代价是每次多一次嵌入调用(约几十毫秒), 相比 LLM 调用可以忽略;
        换来的是拒答闸门在重排器不可用时依然有效。

        **刻意不套时效过滤**（与 retrieve_raw 不同）：
          1. 闸门问的是"这个问题属不属于本知识库的**领域**"，而不是"属不属于现行版本"。
             一个问旧法条的问题仍然是法律领域问题，不该因此被拒。
          2. 阈值 0.26 / 0.30 是在**未过滤**的语料上标定的。换掉候选集会改变分数分布，
             等于让标定作废 —— 要改就得重跑 `scripts/calibrate_domain_gate.py`。
        """
        try:
            nodes = self._vector_retriever.retrieve(query)
        except Exception as exc:  # noqa: BLE001 - 判据失效不应该让整个请求失败
            logger.warning("[HybridRetriever] 领域判据获取失败, 跳过拒答闸门: %s", exc)
            return 1.0  # 拿不到信号就放行, 避免误杀正常问题
        if not nodes:
            return 0.0
        return float(nodes[0].score or 0.0)

    def retrieve_raw(self, query: str) -> List[NodeWithScore]:
        """只做多路召回 + RRF 融合, **不做重排**。

        调用方拿到候选后统一重排一次即可(见 rerank)。
        """
        if not query or not query.strip():
            return []

        vector_nodes = self._vector_retrieve(query)
        logger.debug("[HybridRetriever] 向量召回 %d 条", len(vector_nodes))

        bm25_nodes: List[NodeWithScore] = []
        if self._bm25 is not None:
            bm25_nodes = self._bm25_retrieve(query, top_k=settings.BM25_TOP_K)
            logger.debug("[HybridRetriever] BM25 召回 %d 条", len(bm25_nodes))
        else:
            logger.warning("[HybridRetriever] BM25 未就绪, 本次仅用向量召回")

        fused = self._reciprocal_rank_fusion(vector_nodes, bm25_nodes)
        logger.debug("[HybridRetriever] RRF 融合后 %d 条", len(fused))
        return fused

    # ------------------------------------------------------------
    # 检索层过滤（T-27 / T-28）
    # ------------------------------------------------------------
    def _filter_conditions(self) -> List[tuple[str, str]]:
        """当前生效的元数据过滤条件 `[(字段, 期望值)]` —— **唯一来源**。

        目前只有一条：**排除非现行有效的版本**（需求 F4.2）。

        为什么必须过滤，而不是让模型自己忽略旧条文：
            语料里同一部法有两个版本（劳动法 2009/2018、社保法 2010/2018），
            同一条条文会有两条记录（例如劳动法第二十一条，两条内容还完全一样）。
            靠提示词让 LLM"别看废止的"是概率保证；在这里过滤是**确定性保证**。

        为什么必须在**取 top-k 之前**过滤：
            top-k 的坑位有限。先召回再筛，等于让旧版本占掉一半坑位，
            有效版本被挤出去 —— 召回率会掉，而且掉得不明显（只是"排得靠后了"）。

        两条检索路径共用这份条件：
            - **向量（Milvus）**：llama-index 会把它翻成 Milvus 的 `expr` **下推**（真过滤）
            - **BM25（内存索引）**：内存索引没有查询层，只能在打分后、取 top-k 前筛
            共用同一份条件，避免"两条路各写一套过滤规则"然后慢慢长歪。

        **每次都读 settings 而不是构造时读一次** —— 开关改完立刻生效，不用重启
        （对应需求 F6.3「开关必须在服务端生效且能被验证」）。
        """
        if not settings.ENABLE_VALIDITY_FILTER:
            return []
        return [("validity_status", "effective")]

    def _build_metadata_filters(self) -> Optional["MetadataFilters"]:
        """把过滤条件下推给 Milvus（`MetadataFilters` → `expr`）。"""
        conditions = self._filter_conditions()
        if not conditions:
            return None
        from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

        return MetadataFilters(
            filters=[MetadataFilter(key=key, value=value) for key, value in conditions]
        )

    def passes_filter(self, metadata: dict) -> bool:
        """内存索引（BM25）那一路用的谓词 —— 与 Milvus 那份条件同源。"""
        return all(metadata.get(key) == value for key, value in self._filter_conditions())

    def _vector_retrieve(self, query: str) -> List[NodeWithScore]:
        """向量召回（带元数据过滤）。

        `as_retriever` 每次调用会新建一个轻量对象（就是这个 retriever 本身），
        开销可忽略；换来的是"过滤条件可以随配置变化"。
        """
        retriever = self._index.as_retriever(
            similarity_top_k=self._vector_top_k,
            filters=self._build_metadata_filters(),
        )
        return retriever.retrieve(query)

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """对候选统一精排, 返回的分数已归一化到 [0,1]。"""
        top_n = top_n or settings.RERANK_TOP_N
        if not nodes:
            self._last_rerank_ok = False
            return []
        if not self._use_reranker:
            logger.debug("[HybridRetriever] 重排已关闭, 直接取前 %d 条", top_n)
            self._last_rerank_ok = False
            return nodes[:top_n]
        return self._rerank(query, nodes, top_n=top_n)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[NodeWithScore]:
        """完整检索: 召回 -> 重排 -> 阈值过滤。

        阈值作用在**归一化后**的相关性分上(见 normalize_rerank_score)。
        """
        if not query or not query.strip():
            return []

        top_k = top_k or settings.TOP_K
        similarity_threshold = (
            settings.SIMILARITY_THRESHOLD
            if similarity_threshold is None
            else similarity_threshold
        )

        fused = self.retrieve_raw(query)
        if not fused:
            return []
        if not self._use_reranker:
            # 关闭重排时不做阈值过滤 —— RRF 分数是"多路排名倒数之和"(量级 0.01 上下),
            # 不是相似度。拿相似度阈值去卡它会把结果全部误杀, 这是最隐蔽的一类 bug:
            # 系统不报错, 只是永远返回空。
            logger.debug(
                "[HybridRetriever] 重排未启用, 直接返回融合结果前 %d 条(不套相似度阈值)", top_k
            )
            return fused[:top_k]

        reranked = self.rerank(query, fused, top_n=max(top_k, settings.RERANK_TOP_N))

        if not self._last_rerank_ok:
            # 重排"配置上开着"但**实际没生效**(模型加载失败 / 打分抛异常)。
            # 这与"重排被关闭"是同一个陷阱, 只是更隐蔽: 一旦在这里照常套阈值,
            # RRF 分数(0.03 量级)会被 0.1 全部误杀 —— 表现为"系统什么也查不到",
            # 而日志里只有一行 warning。加载失败在本机是真实场景(内存不足)。
            logger.warning(
                "[HybridRetriever] 重排未生效(模型未加载或执行失败), 当前分数为 RRF 原值, "
                "跳过相似度阈值(%.3f), 直接返回前 %d 条",
                similarity_threshold,
                top_k,
            )
            return reranked[:top_k]

        if similarity_threshold and similarity_threshold > 0:
            kept = [node for node in reranked if (node.score or 0.0) >= similarity_threshold]
            if not kept and reranked:
                # 阈值把结果全过滤掉通常是配置问题, 必须留痕才好发现
                logger.warning(
                    "[HybridRetriever] 阈值 %.3f 过滤掉了全部 %d 条候选(最高分 %.4f), "
                    "请检查阈值语义或重排器分数分布",
                    similarity_threshold,
                    len(reranked),
                    max((n.score or 0.0) for n in reranked),
                )
            reranked = kept

        return reranked[:top_k]

    # ------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------
    def _bm25_retrieve(self, query: str, top_k: int = 10) -> List[NodeWithScore]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize_zh(query))
        if len(scores) == 0:
            return []

        # 过滤必须在取 top-k **之前**做：内存索引没有查询层可下推，
        # 若等取完 top-k 再筛，坑位已经被过滤掉的条目占住了（见 _filter_conditions 的说明）。
        eligible = [self.passes_filter(node.metadata or {}) for node in self._nodes]
        if not all(eligible):
            scores = np.where(eligible, scores, -np.inf)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[NodeWithScore] = []
        for index in top_indices:
            score = float(scores[index])
            if score > 0 and index < len(self._nodes):
                results.append(NodeWithScore(node=self._nodes[index], score=score))
        if not results:
            # 关键词这一路空手而归必须留痕: 否则混合检索会静默退化成纯向量检索,
            # 表现为"召回质量变差"却查不出原因。
            logger.warning(
                "[HybridRetriever] BM25 无命中(语料 %d 条, 最高原始分 %.4f), "
                "本次混合检索实际退化为纯向量召回",
                len(self._nodes),
                float(max(scores)) if len(scores) else 0.0,
            )
        return results

    # ------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------
    def _reciprocal_rank_fusion(
        self,
        *retriever_results: List[NodeWithScore],
        k: int = 60,
    ) -> List[NodeWithScore]:
        """RRF(d) = Σ 1 / (k + rank_i(d))

        注意: 融合后 score 的含义是"多路排名倒数之和", **不是相似度**。
        后续重排会把它替换掉, 所以不要把 RRF 分数当作相关性分使用。
        """
        node_scores: dict[str, float] = {}
        node_map: dict[str, TextNode] = {}

        for results in retriever_results:
            for rank, node_with_score in enumerate(results):
                node_id = node_with_score.node.node_id
                if node_id not in node_scores:
                    node_scores[node_id] = 0.0
                    node_map[node_id] = node_with_score.node
                node_scores[node_id] += 1.0 / (k + rank + 1)

        ordered = sorted(node_scores.items(), key=lambda item: item[1], reverse=True)
        return [NodeWithScore(node=node_map[node_id], score=score) for node_id, score in ordered]

    # ------------------------------------------------------------
    # 重排器
    # ------------------------------------------------------------
    def preload_reranker(self) -> bool:
        """启动期预热重排器。

        原实现是在请求路径上懒加载, 第一个用户要等约 1GB 模型下载/加载。(T-11)
        预热失败不阻塞服务启动, 只把重排降级掉。
        """
        if not self._use_reranker:
            return False
        self._load_reranker()
        return self._reranker is not None

    @property
    def reranker_ready(self) -> bool:
        return self._reranker is not None

    def _load_reranker(self) -> None:
        if self._reranker is not None or self._reranker_failed:
            return
        try:
            # 早期实现直接调用 llama_index.core.postprocessor.SentenceTransformerRerank,
            # 曾因类名拼写错误(SentenceTransformerReranker, 多了 er)被 try/except 静默吞掉,
            # 导致重排功能从未真正生效过 —— 这条教训是"加载失败必须 error 级留痕"的由来。
            # 现在改用自研的 _CrossEncoderReranker 封装(接口不变), 额外支持
            # RERANK_TORCH_DTYPE / RERANK_BACKEND 两个降低内存占用的旋钮(D-2, N-9)。
            self._reranker = _CrossEncoderReranker(
                model=settings.RERANK_MODEL,
                top_n=settings.RERANK_CANDIDATE_LIMIT,
                backend=settings.RERANK_BACKEND,
                torch_dtype=settings.RERANK_TORCH_DTYPE,
                keep_retrieval_score=True,
            )
            logger.info(
                "[HybridRetriever] 重排器加载完成: %s (backend=%s, dtype=%s)",
                settings.RERANK_MODEL,
                settings.RERANK_BACKEND,
                settings.RERANK_TORCH_DTYPE,
            )
        except Exception as exc:  # noqa: BLE001
            self._reranker_failed = True
            logger.error(
                "[HybridRetriever] 重排器加载失败, 检索将退化为无重排: %s", exc, exc_info=True
            )

    def _check_score_semantics(self, raw_scores: List[float]) -> None:
        """一次性校验重排器分数的量纲是否与配置一致。

        如果配置说"这是概率"但实际返回了 [0,1] 之外的值, 说明它其实是 logits ——
        此时阈值语义是错的。必须让使用者看见, 而不是静默钳制成 0/1。
        """
        if self._score_semantics_checked or not raw_scores:
            return
        self._score_semantics_checked = True
        if settings.RERANK_SCORE_IS_LOGIT:
            logger.info("[HybridRetriever] 重排分数按 logits 处理(已施加 sigmoid)")
            return
        out_of_range = [score for score in raw_scores if score < 0.0 or score > 1.0]
        if out_of_range:
            logger.error(
                "[HybridRetriever] 重排器返回了 [0,1] 之外的分(样例 %s), 说明它输出的是 logits; "
                "当前 RERANK_SCORE_IS_LOGIT=False 会把它们钳制到 0/1, 使阈值 %.2f 失去意义。"
                "请把 settings.RERANK_SCORE_IS_LOGIT 设为 True。",
                [round(score, 3) for score in out_of_range[:3]],
                settings.SIMILARITY_THRESHOLD,
            )

    def _rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: int = 5,
    ) -> List[NodeWithScore]:
        if not nodes:
            self._last_rerank_ok = False
            return []
        self._load_reranker()
        if self._reranker is None:
            self._last_rerank_ok = False
            return nodes[:top_n]

        try:
            raw_ranked = self._reranker.postprocess_nodes(nodes, query_str=query)
        except Exception as exc:  # noqa: BLE001
            logger.error("[HybridRetriever] 重排执行失败, 退回召回顺序: %s", exc)
            self._last_rerank_ok = False
            return nodes[:top_n]

        normalized: List[NodeWithScore] = []
        raw_scores: List[float] = []
        for node_with_score in raw_ranked:
            raw_score = float(node_with_score.score) if node_with_score.score is not None else 0.0
            raw_scores.append(raw_score)
            node = node_with_score.node
            node.metadata["rerank_raw"] = raw_score
            normalized.append(NodeWithScore(node=node, score=normalize_rerank_score(raw_score)))

        self._check_score_semantics(raw_scores)

        normalized.sort(key=lambda item: item.score or 0.0, reverse=True)
        self._last_rerank_ok = True
        logger.info(
            "[HybridRetriever] 重排完成: 候选 %d 条, 归一化后 top3 = %s",
            len(normalized),
            [round(n.score or 0.0, 4) for n in normalized[:3]],
        )
        return normalized[:top_n]

    # ------------------------------------------------------------
    # 上下文与引用(委派给唯一事实来源)
    # ------------------------------------------------------------
    def get_context_text(self, nodes: List[NodeWithScore]) -> str:
        from app.core.generate.citation import build_context_block

        return build_context_block(nodes)

    def get_sources_info(self, nodes: List[NodeWithScore]) -> List[dict]:
        from app.core.generate.citation import build_sources

        return build_sources(nodes)
