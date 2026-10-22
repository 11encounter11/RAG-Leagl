# 个人法律知识问答助手 (Legal RAG Assistant)

> 一个**单机自用**的法律知识问答系统：用大白话提问，得到「每一条依据都能点开核验原文、并且标注现行是否有效」的回答；查不到就明确拒答，不编法条。

差异化不在"回答更聪明"，在**回答更可信**。整个系统围绕一件小事反复打磨：**让每一条引用都能被机器验证为真实存在于知识库中的现行条文**。

⚠️ **免责声明**：本项目定位是**法律信息查询与普法**，只回答"法律怎么规定的、依据是哪一条"，**不出具法律意见、不判定输赢、不预测判决结果**。涉及自身权益的具体处置，请咨询执业律师或当地主管部门（依据《律师法》第十三条）。

**目录**：[要解决的真实问题](#一要解决的真实问题) · [核心特性](#二核心特性) · [实测数字](#三实测数字) · [技术栈](#四技术栈) · [怎么跑起来](#五怎么跑起来) · [快速开始](#六快速开始) · [日常使用](#七日常使用) · [配置速查](#八配置速查) · [评测](#九评测) · [项目结构](#十项目结构) · [踩坑 FAQ](#十一踩坑记录faq) · [已知限制](#十二已知限制) · [贡献与许可](#十三贡献与许可)

各目录另有说明文件：[`app/README.md`](app/README.md)（代码地图）、[`scripts/README.md`](scripts/README.md)（6 个脚本各自何时该跑）、[`sql/README.md`](sql/README.md)（建表与 5 张表谁在读写）、[`data/README.md`](data/README.md)（语料怎么来、评测集怎么用）。

---

## 一、要解决的真实问题

通用大模型在法律场景有三个**它自己解决不了**的问题：

| 问题 | 表现 | 本项目的应对 |
|---|---|---|
| **不可核验** | 引用的法条无法确认是否真实存在、是否被曲解。已有实证评估显示主流法律 AI 工具准确率 19%~65%、幻觉率 17%~33% | 每条引用带 `法名 + 条文号 + 版本 + 时效`，可点开看逐字原文；生成后再过一遍**引用白名单校验**，权威条文表里查不到的一律标"未能核验" |
| **时效错乱** | 会引用已废止的条文，而且语气笃定 | 时效过滤下推到 **Milvus 查询表达式**（`validity_status == "effective"`），旧版本根本进不了候选池——靠提示词让模型"别看废止条文"只是概率保证，查询层过滤是确定性保证 |
| **不知道你的数据** | 你手上那份文件、某个省市的地方性法规，它一无所知 | 文档摄取管线 + 按条切分 + 26 字段元数据契约，本地随时扩充语料 |

法律恰恰是**零容错**领域：错误的法条引用比"不知道"危害大得多。所以工程优先级是 **可核验 > 拒答准确 > 召回质量 > 回答文采**。

---

## 二、核心特性

**检索链路**

- **混合检索**：Milvus 向量召回 + BM25 关键词召回（jieba 分词），RRF 融合，再用 Cross-Encoder（`BAAI/bge-reranker-base`）本地重排。
- **元数据过滤**：按知识库、地域、时效状态、文档类型过滤。问句点名某部法时（例如"劳动法第四十七条讲了什么"），`law_name` 条件下推到向量与 BM25 两路，避免它被**同条号**的《劳动合同法》挤掉——Cross-Encoder 认条号不认法名，这件事只能靠过滤解决。
- **独立拒答闸门**：判据是**向量原始相似度的最高值**，不经过 RRF、不依赖重排器。为什么必须独立——重排器是可选组件，可能因内存不足加载失败；把安全底线建立在可选组件上，实测结果是"该拒答的一个都不拒"（拒答准确率 0%）。
- **Agentic 工作流**（LangGraph，9 节点）：查询改写 → 多路召回 → 领域闸门 → 重排 → 反思打分 → 放行 / 换参数重试 / 拒答。重试**不回**改写节点（改写 prompt 没变，重跑没意义），只换检索参数：`top_k` 放大 / 阈值下调 / 拆子问题。
- **引用白名单校验**：回答里每条法条引用都必须在 `law_articles`（条文权威表）里查得到。查不到不删答案，而是在回答上明确标注"⚠️ 部分依据未能核验"——**不隐瞒，也不丢掉答案**。

**数据与可复现性**

- **按条切分 + 策略注册表**：`statute`（法律法规，按条切）/ `faq`（一问一块）/ `generic`（兜底）。解析失败按「结构感知 → 递归分隔符 → 定长」降级，**永不返回空结果**，并留痕实际用的策略与置信度。
- **26 字段元数据契约**：chunk 携带法名、条文号（原文 + 归一化）、章节路径、版本 ID、生效/失效日期、时效状态、地域、切分策略等，与 Milvus 标量列一一对应。
- **版本对**：同一部法的多个版本共存（《劳动法》2009 修正 + 2018 修正），旧版标 `amended` 默认被过滤；**关掉过滤仍能召回**，用来证明"过滤确实生效"而不是"数据压根没有"。
- **删除即清理**：删文档会同时删掉 Milvus 里对应向量。这里踩过坑——框架曾把 `document_id` 静默覆盖成字符串 `"None"`，导致按 ID 删向量永远匹配不到、删完还显示成功。
- **每次问答可复盘**：`messages.retrieval_info`（JSON）记下实际生效的阈值、改写句、反思的结构化判断、每条候选的重排分。事后能回答"这次为什么答错了"。

---

## 三、实测数字

语料：9 份官方 docx = 7 部法 / **703 个条文片段**；评测集 **52 题**（36 道可判定 + 16 道超纲），见 [`data/eval/`](data/eval/README.md)。

**direct 基线**（单次混合检索，重排开、时效过滤开，`data/eval/baseline.json`）

| Recall@5 | 完整召回@5 | MRR | NDCG@5 | 旧版混入率 | 未命中 |
|---|---|---|---|---|---|
| **0.9167** | 0.8056 | 0.7315 | 0.7612 | **0.0000** | 3 条 |

**agentic 基线**（生产路径：改写 → 召回 → 重排 → 反思 → 重试/拒答，`data/eval/baseline-agentic.json`）

| 端到端 Recall@5 | 完整召回 | MRR | NDCG@5 | 放行子集 Recall@5 | 库内误拒 | 超纲拒答 |
|---|---|---|---|---|---|---|
| **0.8611** | 0.8056 | 0.7775 | 0.7605 | 0.9118 (n=34) | 2/36 | **16/16 = 1.0** |

这两组数字**可以自己复现**：`python scripts/run_eval.py --top-k 5`（direct）与 `--mode agentic`，需要中间件起着、API 额度可用。

> **两条路径必须分开看。** 生产走 agentic，而评测默认走 direct；同一处修复可能在一条链上生效、另一条链上完全测不到——本项目就发生过"评测全绿、生产把所有正常问题都拒答"的事故。
> 另外 **agentic 的端到端 recall 不能与 direct 相减**：工作流一旦拒答就清空候选、该题在指标里记 0，于是"没检到"和"检到了但反思不放行"会混成同一个数。

---

## 四、技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| Web 框架 | FastAPI + Uvicorn | REST + SSE 流式输出 |
| RAG 框架 | LlamaIndex | 文档加载 / 向量化 / Milvus 集成 |
| 工作流 | LangGraph | Agentic 检索的 9 节点状态机 |
| 生成模型 | DeepSeek（`deepseek-v4.1-flash`） | 改写、反思、生成统一用一个 |
| 向量模型 | 智谱 `embedding-3`（1024 维） | API 调用，无需本地下载模型 |
| 重排模型 | `BAAI/bge-reranker-base` | Cross-Encoder，**本地 CPU 推理**，float32 约 1 GB 常驻 |
| 向量库 | Milvus 2.4 standalone（+ etcd + MinIO） | 26 个标量字段，支持查询表达式过滤 |
| 元数据/历史 | MySQL 8.0（5 张表） | 见 [`sql/README.md`](sql/README.md) |
| 缓存 | Redis 7 | 查询改写与反思结果缓存——让同题两次跑出一致结果，评测才可复现 |
| 前端 | Streamlit 单文件 | 上传、Top-K/阈值调节、流式问答、会话管理 |

---

## 五、怎么跑起来

### 5.1 入库链路

```
DOCX/PDF/MD/TXT 上传（或 index-dir 批量导入）
  ├─ 扩展名校验 → 落盘 data/raw/
  ├─ 内容 SHA-256 去重（重复上传返回 409，不重复入库）
  ├─ loader 多格式读取
  ├─ resolve_ingestor：①用户指定 doc_type ②结构探测器 ③文件名规则 ④generic 兜底
  ├─ 按条切分 → 若干 section（解析不出结构则降级，留痕 fallback_from）
  ├─ 拼 26 字段元数据
  │    ├─ 章节路径拼进 node.text（只用于向量化，提升召回）
  │    └─ display_text 另存干净原文（用于引用展示，两者刻意分开）
  ├─ 智谱 embedding → 写 Milvus（维度自动探测，建库前校验）
  ├─ 写 MySQL documents / law_articles（条文权威源）
  └─ 重建 BM25 语料
```

### 5.2 问答链路

```
POST /api/chat
  ├─ 会话解析（无则创建）
  ├─ 检索分支：ENABLE_AGENTIC_RAG ∧ 请求参数 use_agentic_rag
  │    ├─【工作流】analyze_query(改写) → generate_hyde(默认关) → hybrid_retrieve
  │    │            → assess_domain(拒答闸门) → rerank → reflect
  │    │                ├─ generate → END
  │    │                ├─ retry  → 换参数回到 hybrid_retrieve（不回改写节点）
  │    │                └─ refuse → END
  │    └─【纯检索】向量 + BM25 → RRF → 重排 → 阈值准入
  ├─ 上下文组装（每条带 [法名·条文号·时效状态] 前缀）
  ├─ 生成（三段式：结论 / 依据 / 行动建议）
  ├─ 引用白名单校验（权威表里查不到的标注"未能核验"）
  └─ 落库 messages（retrieval_info / tokens_used / duration_ms）
```

逐模块的代码地图见 [`app/README.md`](app/README.md)。

---

## 六、快速开始

### 0. 前置要求

| 项 | 要求 |
|---|---|
| Python | 3.11+ |
| Docker | 跑 5 个容器（MySQL + Redis + Milvus + etcd + MinIO） |
| 内存 | 中间件 + 后端进程合计约 **3~4 GB**；重排器 float32 独占约 1 GB |
| API Key | 智谱开放平台（供 `embedding-3`）+ DeepSeek（供改写/反思/生成），两家都按用量计费 |
| 磁盘 | Milvus + MySQL 数据卷约需 5 GB |

> 重排器加载约需 30~35 秒，且是在**启动期预热**的（不是第一次请求时才加载）。看到日志 `系统启动完成` 之前多等一会儿是正常的。

### 1. 克隆并装依赖

```bash
git clone <仓库地址> legal-rag
cd legal-rag

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> `requirements.txt` 里几个依赖看着"多余"但**不能删**，每一条旁边都写了理由：`streamlit`（前端入口）、`jieba`（缺失时 BM25 会**静默降级**成字符 bigram，不报错、只是质量变差）、`torch`（`retriever.py` 直接按 dtype 分支 import 它）。

### 2. 起中间件

```bash
docker compose up -d
docker compose ps        # 5 个容器都要 Up；milvus 依赖 etcd 与 minio，首次启动约 1~2 分钟
```

MySQL 首次启动会自动执行 `sql/init.sql` 建好 5 张表。这个初始化**只在数据卷为空时发生一次**——改过 `init.sql` 需要先 `docker compose down -v`（会清空数据卷）再 `up -d`。细节见 [`sql/README.md`](sql/README.md)。

### 3. 配置

```bash
cp .env.example .env
# 至少填这两个
#   ZHIPU_API_KEY=...      # 智谱，供 embedding-3
#   DEEPSEEK_API_KEY=...   # DeepSeek，供改写/反思/生成
```

`.env` 已在 `.gitignore` 里。提醒一句：**git 历史是永久的**，密钥一旦提交过，后来删掉文件也清不干净，只能重写历史并立刻去服务商后台轮换。

### 4. 准备语料（必读）

`data/raw/` 被忽略，**clone 下来是空的**，需要你自己获取法规原文：

- **来源**：国家法律法规数据库 `https://flk.npc.gov.cn/` → 站内搜索 → 勾选条目 → 「批量下载文件」，选 `.docx`（官方排版正文）。
- **不要用 AI 生成的文本当底本**：本项目的立身之本是"每条依据可核验"，底本来源说不清，这个承诺从根上不成立。
- **命名规则**：`<法名>_<公布或修正日期YYYYMMDD>.docx`，**日期后缀必须保留**。版本信息按文件名匹配清单，名字错了 → 时效状态拿不到 → 时效过滤静默失效。
- 完整下载步骤、当前 9 份文件清单与哈希校验，见 [`data/README.md`](data/README.md)。

```bash
python scripts/verify_corpus.py     # 核对下没下错、有没有改动
```

> 想复现上面那两组指标，就按清单下同样的 9 份文件。只放一两部法也能跑通流程，但**数字不成立**。

### 5. 灌版本数据 → 起服务 → 索引语料

```bash
# ① 语料清单里的版本信息写进 law_versions（幂等，可反复执行）
python scripts/seed_law_versions.py --dry-run
python scripts/seed_law_versions.py

# ② 启动后端（首次加载重排器，约 30~35 秒）
python app/main.py
#   或开发模式： uvicorn app.main:app --reload --port 8000

# ③ 健康检查：mysql / redis / milvus 三个都应为 true
curl http://localhost:8000/api/health
```

另开一个终端，索引 `data/raw/` 下全部文档：

```bash
curl -X POST "http://localhost:8000/api/knowledge/index-dir?dir_path=data/raw&doc_type=statute&kb_id=default"

# 确认入库结果：documents 有行、law_articles 应有 703 行、Milvus 集合有向量
python scripts/probe_vector_store.py
```

起前端：

```bash
streamlit run streamlit_app.py    # 默认 http://localhost:8501
```

### 6. 问一句试试

> 我在公司干了两年三个月，被辞退能拿多少补偿？

期望：给出「N 个月工资」的结论 + **《劳动合同法》第四十七条**（条文号正确）+ 行动建议；点开引用能看到逐字原文与"2012 年修正 · 现行有效"标注。

再问一句超纲的（"火星上的房产税怎么交"），期望是**明确拒答且不出现任何法条编号**。这两条就是本项目最重要的验收动作。

---

## 七、日常使用

### 7.1 Web UI

侧边栏可调 Top-K、准入阈值、上传文档、查看/删除会话。界面上传开放 `pdf/docx/md/txt`，后端 `POST /api/knowledge/upload` 还支持 `xlsx/csv`；批量导入走 `index-dir` 或直接调 API。

### 7.2 HTTP API

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/api/chat` | 非流式问答，返回 `content` / `sources` / `workflow_steps` / `domain_score` / `refused` / `confidence_level` |
| `POST` | `/api/chat/stream` | SSE 流式，事件类型 `token` / `sources` / `done` / `error` |
| `GET` | `/api/conversations` | 会话列表 |
| `GET` | `/api/conversations/{id}/messages` | 会话历史消息 |
| `PUT` | `/api/conversations/{id}` | 重命名会话 |
| `DELETE` | `/api/conversations/{id}` | 删除会话（级联删消息） |
| `POST` | `/api/knowledge/upload` | 上传单文档（`multipart`，可带 `doc_type` / `kb_id`） |
| `POST` | `/api/knowledge/index-dir` | 批量索引目录 |
| `GET` | `/api/knowledge/documents` | 文档列表（含实际切分策略、块数、状态） |
| `DELETE` | `/api/knowledge/documents/{id}` | 删除文档 **+ 同步删向量** |
| `GET` | `/api/knowledge/chunks/{chunk_id}` | 回查片段原文（"引用可核验"的落地接口） |
| `GET` | `/api/health` | MySQL / Redis / Milvus 连通性 |
| `GET` | `/api/models` | 可用模型列表 |

交互式文档：<http://localhost:8000/docs>。

```bash
curl -s http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"公司三个月没发工资，我可以怎么做？","model_name":"deepseek-v4.1-flash","top_k":5,"use_agentic_rag":true}' \
  | python -m json.tool
```

`sources[].law_name / article_no / validity_label / version_label / score_normalized` 就是每条依据的完整出处。`workflow_steps` 是**带参数的节点名**，能直接读出这次跑了什么：`query_analysis:cache`（改写命中缓存）、`law_filter(...)`、`domain_gate(0.4123->high)`、`rerank`、`reflection(0.86)`、`retry#1({...})`、`refuse(原因)`。

更细的复盘线索在 **MySQL `messages.retrieval_info`**（JSON 列）：实际用于检索的改写句 `rewritten_query`、反思的结构化判断（`core_basis` / `coverage` / `missing`）、以及 `threshold_usage`（这次阈值到底比了什么、top1 多少、保留了几条）。

### 7.3 换语料的正确顺序（不能反）

```bash
python scripts/verify_corpus.py       # ① 清单与文件是否一致
python scripts/reset_vector_store.py  # ② 清掉旧向量（先 --dry-run；删文件 ≠ 删向量）
python scripts/seed_law_versions.py   # ③ 版本数据
# ④ 重新索引： curl -X POST ".../api/knowledge/index-dir?dir_path=data/raw&doc_type=statute"
python scripts/probe_vector_store.py  # ⑤ 确认进了
python scripts/run_eval.py --check    # ⑥ 标准答案是否仍在语料里
python scripts/run_eval.py --top-k 5  # ⑦ 与基线对比
python scripts/calibrate_domain_gate.py  # ⑧ **重标拒答阈值**（语料一变就必须做）
```

---

## 八、配置速查

全部配置在 `app/config/settings.py`（每个不直观的取值旁边都写了为什么），从 `.env` 读，`.env.example` 里每个键都有注释。这里只列**语义容易搞错**的几个：

| 键 | 默认 | 真实语义 |
|---|---|---|
| `SIMILARITY_THRESHOLD` | 0.1 | 名字叫"相似度阈值"，实际比的是**重排归一化分**（0~1，不是余弦相似度）；且只看本批**第 1 名**：top1 达标就保留整批，最终条数交给 `TOP_K`。它是"这批值不值得答"的粗闸门，**不是每条片段的及格线**。数值来自评测集标定，语料/切分一变就不可比 |
| `DOMAIN_SCORE_REJECT` / `DOMAIN_SCORE_CAUTION` | 0.345 / 0.36 | 拒答闸门，判据是**向量原始相似度最高值**（与上面的重排分是两个量纲）。用 `scripts/calibrate_domain_gate.py` 跑出库内/超纲两分布后选点。当前两分布之间的**空档只有 0.0105**，换嵌入模型、扩语料、加超纲题都会让它静默失效（表现只是"超纲题又被回答了"） |
| `ENABLE_VALIDITY_FILTER` | true | 开：`validity_status == "effective"` 下推进 Milvus 查询表达式，旧版本进不了候选。关：旧版也能召回——这是**对照组**，用来证明过滤真的生效，而不是数据本来就没有 |
| `ENABLE_AGENTIC_RAG` | true | 关掉后工作流**真的不执行那些节点**（服务端日志与 `workflow_steps` 可见），不是"前端不传参"那种伪生效 |
| `RERANK_TORCH_DTYPE` | float32 | 内存吃紧时的第一档缓解手段（`float16` / `bfloat16` 把权重内存砍半）。CPU 上 float16 部分算子支持有限，报错就退回 float32 或改用 bfloat16 |
| `HF_HUB_OFFLINE` | true | 默认只读本地模型缓存。设为 false 才会去 huggingface.co 下载（首次装重排器时临时改） |
| `EMBEDDING_MODEL_NAME` | embedding-3 | **死配置**：没有任何代码读它。实际模型名在 `app/core/rag/embedder.py` 里硬编码。改它不会生效 |

---

## 九、评测

```bash
python scripts/run_eval.py --check                    # 只自检标准答案是否真在语料里，不连任何中间件
python scripts/run_eval.py --top-k 5                  # direct：纯检索。每题一次查询嵌入
python scripts/run_eval.py --top-k 5 --mode agentic   # agentic：经改写/重排/反思，会调 LLM（但不生成答案）
python scripts/run_eval.py --top-k 5 --mode gate      # 拒答闸门专项（超纲拦住多少、库内误杀多少）
python scripts/run_eval.py --top-k 5 --mode agentic --limit 3   # 冒烟；冒烟报告不能当基线
```

报告落 `data/eval/reports/run-<mode>-<时间戳>.json`，含**逐题明细**：召回了什么、每条重排分、改写句、反思分、重试次数、阈值砍掉哪几条。指标口径与匹配规则的完整说明在 [`data/eval/README.md`](data/eval/README.md)。

评测按「**法名 + 条文号**」匹配，不让 LLM 当裁判。法条编号是确定性的，chunk 元数据里就带着，直接比对即可；而 LLM 裁判自身不稳定、而且贵。语料含多部法之后还必须带法名——每部法都有"第三十六条"，只比条文号会把命中《劳动合同法》第三十六条判成命中《劳动法》第三十六条，**指标虚高且不报错**。

> **一条纪律**：不要为了让数字好看而回改标准答案。本项目有一道社保题改判后仍未命中，探针测出同一句口语问法下目标条文排候选池第 11、换成法言法语排第 1——那是真实的检索缺口。把尺子改弯去迁就检索，等于把问题抹掉。

---

## 十、项目结构

```
.
├── app/                    # 后端本体，地图见 app/README.md
│   ├── main.py             # FastAPI 入口（启动期重建 BM25 + 预热重排器）
│   ├── config/settings.py  # 全部配置与默认值
│   ├── core/               # ingest / rag / workflow / generate / eval / llm / memory
│   ├── routers/            # chat / knowledge / conversation
│   └── services/           # 编排层（rag_service 是入库与检索的主干）
├── scripts/                # 6 个命令行入口，见 scripts/README.md
├── sql/
│   └── init.sql            # 5 张表，见 sql/README.md
├── data/                   # 语料与评测集，见 data/README.md
│   ├── raw/                # 语料 docx（gitignore，需自行下载官方文本）
│   ├── corpus_manifest.json# 语料清单 + SHA-256（运行时依赖）
│   └── eval/               # 评测集 + 两份基线 + 口径说明
├── streamlit_app.py        # 前端单文件
├── docker-compose.yml      # MySQL + Redis + Milvus(+etcd+MinIO)
├── requirements.txt
├── .env.example            # 配置模板（键名与 settings.py 一致）
└── .gitignore
```

> 代码注释里偶尔出现 `docs/0X`、`T-26`、`P2-04` 这类引用，那是开发过程文档与任务编号，未随本仓库发布；逻辑本身在代码里是完整可读的。

---

## 十一、踩坑记录（FAQ）

**Q：启动卡在"加载重排器"好几分钟，是不是 OOM 了？**
多半是网络。sentence-transformers 加载**本地已缓存**的模型时，仍会先向 huggingface.co 发 HEAD 请求核对更新；国内网络下该请求超时，然后按 1s/2s/4s… 退避重试 5 次，表现为"卡死几分钟"。默认 `HF_HUB_OFFLINE=true` 就是绕开它（代价：模型没缓存时直接报错而不是自动下载，首次下载时临时设 `false`）。

**Q：OOM / 内存不够？**
`bge-reranker-base` 以 float32 加载约 1 GB 常驻。把 `RERANK_TORCH_DTYPE` 改成 `float16` 或 `bfloat16`，权重内存直接砍半。**重排器加载失败不影响正确性**（排序降一档），但会影响拒答——所以拒答闸门刻意用了不依赖重排器的判据。

**Q：`MilvusException: the dim of field data is not equal to schema dim`？**
向量维度不一致。智谱 `embedding-3` 实际返回 **1024** 维，而早期代码硬编码 2048，这个不一致只在真正插入时才暴露——那时已经白跑了几百次嵌入调用。现在 `EMBEDDING_DIM=0` 表示自动探测 + 建库前校验；只有需要跨环境锁死维度时才显式赋值。

**Q：删了 `data/raw/` 下的文件，旧内容还能被检索到？**
正常现象。删文件 ≠ 删向量，旧片段会继续被召回且不报错。按 [7.3 的顺序](#73-换语料的正确顺序不能反)走。删**单个文档**用 `DELETE /api/knowledge/documents/{id}`，它会同步清理向量。

**Q：BM25 关键词那一路好像没起作用？**
检查 `jieba` 装上没。`retriever.py` 在 jieba 缺失时会**静默降级**成字符 bigram——不报错，只是质量变差。

**Q：改了 `.env` 里的 `EMBEDDING_MODEL_NAME` 没生效？**
它本来就是死配置，见[配置速查](#八配置速查)。

**Q：`/api/health` 显示 degraded？**
按返回体里 `mysql` / `redis` / `milvus` 哪个是 false 逐个查容器。Milvus standalone 依赖 etcd 与 MinIO，三个都得是 `Up`；`docker compose ps` + `docker compose logs milvus` 是最快的入口。注意 MySQL 宿主机端口是 **3307**，`.env` 里 `MYSQL_PORT` 要跟着填 3307。

**Q：回答里没有引用，或引用被标成"未能核验"？**
`sources[].article_no` 与 MySQL `law_articles` 表对不上就会出现这个标注（白名单校验查的是条文权威表）。常见原因：语料换过但没重建索引；或文件名不符合 `<法名>_<日期>.docx` 规则导致拿不到 `version_id`。排查入口见 [`app/README.md`](app/README.md) 的入库链路一节和 [`sql/README.md`](sql/README.md)。

---

## 十二、已知限制

写清楚是为了不让人当成隐藏 bug，也标出下一步：

1. **时效管理只到"整部法"级，没有"条文级"失效。** 《劳动法》2018 修正版整部仍现行有效，所以它的**第八十二条（仲裁时效 60 日）仍会被召回**——尽管该条已被《劳动争议调解仲裁法》第二十七条（1 年）取代。文件级过滤抓不到"条文被新法取代"这种情形，这是当前最硬的一处缺口。**在用这套系统做真实决策前，请把它当索引而不是权威。**
2. **拒答闸门用的是绝对余弦阈值，库内与超纲两分布之间空档只有 0.0105。** 换嵌入模型、扩语料、加超纲题都会让它失效，且失效是静默的。更稳的做法是改用相对判据（top1 与 top2 的差值，或对分布做标准化）。
3. **口语化问法召回差于法言法语。** agentic 基线里 `term` 类 18 题端到端 0.9444，`colloquial` 类 14 题 0.8571（12/14）。典型证据：问「公司一直不给我交社保，我该怎么办？」时目标条文排候选池第 11，换成法言法语后同一条排第 1。「口语 → 法律要素映射」这项能力尚未实现（现有改写节点只做通用改写）。
4. **FAQ 文档类型没有真实语料验证。** 解析器有实现，但端到端效果未经数据检验。
5. **配置项在 `.env`（服务端生效、可验证），没有后台管理界面。** 配置落库 + 变更留痕尚未实现。
6. **不做 OCR。** 扫描件 PDF 会被明确拒绝并说明原因，不做静默降级——OCR 的识别错误会污染引用溯源。
7. **单用户、单机。** 没有多租户、登录、RBAC、配额。项目早期按"企业级"设计过，后来明确收缩，理由是这些能力在个人自用场景下没有真实使用者，是纯成本。
8. **改动前请自备回归。** 本仓库不附单元测试；改检索或切分后，用 `scripts/run_eval.py`（direct / agentic / gate 三种模式）对照两份基线，这是当前唯一的回归手段。

---

## 十三、贡献与许可

想改进它的话，动手前先看一眼 [`data/eval/README.md`](data/eval/README.md) 的指标口径，以及这三条纪律：

- 改检索参数、切分策略、阈值、过滤逻辑之后**必须跑评测回归**：`direct` 的改动对照 `data/eval/baseline.json`，工作流的改动对照 `baseline-agentic.json`，**两组数字不可互减**。
- 语料一变，`verify_corpus.py` → `reset_vector_store.py` → 重新索引 → **重标拒答阈值**，顺序不能反。
- 指标退步时先分清是"尺子变了"还是"真的变差了"，再决定要不要动标准答案。默认答案是**不动标注**。

**许可**：**待定** —— 仓库尚未包含 `LICENSE` 文件（初步倾向 MIT，最终由作者确定）。正式开放前需要一并确认：语料的许可边界（法规原文本身不适用著作权法，见《著作权法》第五条），以及第三方模型（`BAAI/bge-reranker-base`、智谱 embedding API、DeepSeek API）的使用条款。

---

如果这个项目对你理解「怎么把 RAG 做到可验证」有一点参考价值，欢迎点个 Star。
