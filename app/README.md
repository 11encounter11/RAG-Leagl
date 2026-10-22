# `app/` 代码地图

分层是硬规矩：**`routers` 只做 HTTP 出入参 → `services` 编排 → `core` 干活**。`core` 不允许反向 import `services`。

## 目录与职责

| 位置 | 干什么 | 改哪类需求来这里 |
|---|---|---|
| `main.py` | FastAPI 入口。`lifespan` 里连 MySQL、建索引、**重建 BM25 语料**、**预热重排器**（约 30~35 秒）。这四件事都在启动期做，不放第一次请求 | 加路由、加启动检查 |
| `config/settings.py` | 全部配置项与默认值，从 `.env` 读。每个非直觉的取值旁边都写了为什么 | 加开关、调阈值 |
| `models/schemas.py` | 请求/响应模型（`ChatRequest`、`ChatResponse`、`SourceInfo`…） | 改接口字段 |
| `routers/chat.py` `knowledge.py` `conversation.py` | 三组接口，共 11 个 | 加接口 |
| `services/chat_service.py` | 一次问答的完整编排：会话 → 检索分支 → 生成 → 引用核验 → 落库留痕 | 改回答行为 |
| `services/rag_service.py` | 入库与检索的主干：去重、切分、写向量、写 `documents`/`law_articles`、重建 BM25 | 改摄取流程 |
| `core/ingest/` | **按文档类型路由切分策略**。`registry.py` 注册表（`statute`/`faq`/`generic`）、`statute.py` 按条切、`detectors.py` 结构探测、`nodes.py` 拼元数据、`metadata.py` 字段契约 | 加一种文档类型 = 加一个解析器 + 注册，不动别的模块 |
| `core/rag/` | `loader.py` 多格式读取、`embedder.py` 智谱嵌入、`vector_store.py` Milvus、`retriever.py` 混合检索 + RRF + 重排 + 阈值准入 | 改检索质量 |
| `core/workflow/rag_workflow.py` | LangGraph 状态机，**9 个节点**：`analyze_query` → `generate_hyde` → `hybrid_retrieve` → `assess_domain` → `rerank` → `reflect` →（`generate` / `retry` / `refuse`） | 改 Agentic 流程 |
| `core/generate/` | `prompt.py` 提示词（三段式与引用格式要求）、`citation.py` **引用格式与白名单校验的唯一事实来源** | 改回答格式 |
| `core/law_versions.py` | 版本层：读 `data/corpus_manifest.json`，按文件名给 chunk 定位 `version_id`，并生成条文主键 | 改时效/版本 |
| `core/memory/` | `db_store.py` MySQL（5 张表全在这里）、`redis_store.py` 缓存（改写与反思结果） | 改表结构 |
| `core/llm/` | `router.py` 模型路由、`deepseek.py` / `glm.py` 适配 | 换模型 |
| `core/eval/` | 评测用：`dataset.py` 读评测集、`corpus.py` 扫 `data/raw` 建"每部法有哪些条文"的索引、`metrics.py` 命中判档与 Recall/MRR/NDCG | 改评测口径 |
| `utils/sse.py` | SSE 事件格式化（就一行格式，不值得引依赖） | — |

## 三条链路，从哪读起

1. **启动** → `main.py` 的 `lifespan` → `rag_service.init_index()` → `vector_store.load_index()` / `rebuild_bm25_corpus()` / `preload_reranker()`
2. **入库** → `routers/knowledge.py` 的 `upload` 或 `index-dir` → `rag_service.index_file()`：去重哈希 → `loader` → `ingest` 解析切分 → `nodes` 拼 26 字段元数据 → `embedder` + `vector_store` 写 Milvus → `db_store` 写 `documents` 与 `law_articles` → 重建 BM25
3. **问答** → `routers/chat.py` → `chat_service.chat()` → 按 `ENABLE_AGENTIC_RAG` 和请求参数选走 `rag_workflow` 还是纯检索 → `generate` → `citation` 核验 → 落 `messages`

想查"这次为什么答错了"：看 MySQL `messages.retrieval_info` 那列（JSON），里面有实际生效的阈值、改写句、反思的结构化判断、以及每条候选的重排分。接口响应里只有 `workflow_steps` 和 `reflection_score` 这几个摘要字段。

## 两个容易踩的设计约定

- **`node.text` 与 `display_text` 是两份东西**。前者拼进了章节路径等结构前缀（只为了向量召回更准），后者是干净的条文原文（给用户看、存进引用）。混用会让展示出来的"法条原文"里夹着章节标题。
- **`document_id` 必须显式设置**。LlamaIndex 在某些路径上会把它覆盖成字符串 `"None"`，后果是"按文档 ID 删向量"永远匹配不到，删完还显示成功，留下孤儿向量。相关回归在 `rag_service` / `vector_store` 里。

> 代码注释里偶尔会出现 `docs/0X-xxx.md`、`T-26`、`P2-04` 这类引用——那是开发过程中的设计文档与任务编号，**未随本仓库发布**。看到就当它不存在，逻辑本身在代码里是完整可读的。
