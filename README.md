# 个人法律知识问答助手 (Legal RAG Assistant)

> 一个**单机自用**的法律知识问答系统：用大白话提问，得到「每一条依据都能点开核验原文、并且标注现行是否有效」的回答；查不到就明确拒答，不编法条。

差异化不在"回答更聪明"，在**回答更可信**。为此整个系统围绕一件小事反复打磨：**让每一条引用都能被机器验证为真实存在于知识库中的现行条文**。

⚠️ **免责声明**：本项目定位是**法律信息查询与普法**，只回答"法律怎么规定的、依据是哪一条"，**不出具法律意见、不判定输赢、不预测判决结果**。涉及自身权益的具体处置，请咨询执业律师或当地主管部门（依据《律师法》第十三条）。

<details>
<summary>English summary</summary>

A single-user Retrieval-Augmented-Q&A system for Chinese labor law. Every citation in an answer is checked against the vector store's whitelist before being returned; expired/repealed versions are filtered out at the **query layer** (Milvus scalar expression), not by asking the LLM to behave; and an out-of-scope question is refused by a gate that deliberately does **not** depend on the reranker. Quality claims are backed by a 52-question evaluation set with checked-in baselines. Stack: FastAPI + LlamaIndex + LangGraph + Milvus + MySQL + Redis, everything in one `docker compose`.

</details>

---

## 目录

- [一、要解决的真实问题](#一要解决的真实问题)
- [二、核心特性](#二核心特性)
- [三、当前实测数字](#三当前实测数字)
- [四、技术栈](#四技术栈)
- [五、它是怎么跑起来的](#五它是怎么跑起来的)
- [六、快速开始](#六快速开始)
- [七、日常使用](#七日常使用)
- [八、配置项速查](#八配置项速查)
- [九、测试与评测](#九测试与评测)
- [十、项目结构](#十项目结构)
- [十一、踩坑记录（FAQ）](#十一踩坑记录faq)
- [十二、已知限制](#十二已知限制)
- [十三、文档索引](#十三文档索引)
- [十四、贡献与许可](#十四贡献与许可)

---

## 一、要解决的真实问题

通用大模型在法律场景有三个**它自己解决不了**的问题：

| 问题 | 表现 | 本项目的应对 |
|---|---|---|
| **不可核验** | 引用的法条无法确认是否真实存在、是否被曲解。已有实证评估显示主流法律 AI 工具准确率 19%~65%、幻觉率 17%~33% | 每条引用都带 `法名 + 条文号 + 版本 + 时效`，可点开看逐字原文；生成后再过一遍**引用白名单校验**，库里没有的条文号一律标为"未能核验" |
| **时效错乱** | 会引用已废止的条文，而且语气笃定 | 时效过滤下推到 **Milvus 查询表达式**（`validity_status == "effective"`），旧版本根本进不了候选池 —— 靠提示词让模型"别看废止条文"只是概率保证，查询层过滤是确定性保证 |
| **不知道你的数据** | 你手上那份文件、某个省市的地方性法规，它一无所知 | 文档摄取管线 + 按条切分 + 26 字段元数据契约，本地随时扩充语料 |

法律恰恰是**零容错**领域：错误的法条引用比"不知道"危害大得多。所以本项目的工程优先级是 **可核验 > 拒答准确 > 召回质量 > 回答文采**。

---

## 二、核心特性

**检索链路**

- **混合检索**：Milvus 向量召回 + BM25 关键词召回（jieba 分词），RRF 融合，再用 Cross-Encoder（`BAAI/bge-reranker-base`）本地重排。
- **元数据过滤**：按知识库、地域、时效状态、文档类型过滤；问句点名某部法时（P2-21），`law_name` 条件下推到向量与 BM25 两路，避免"《劳动法》第四十七条"被同条号的《劳动合同法》挤掉。
- **独立拒答闸门**：判据是**向量原始相似度的最高值**，不经过 RRF、不依赖重排器。为什么必须独立——重排器是可选组件，可能因内存不足加载失败；把安全底线建立在可选组件上，实测结果是"该拒答的一个都不拒"（拒答准确率 0%）。
- **Agentic 工作流**（LangGraph 9 节点）：查询改写 → 多路召回 → 领域闸门 → 重排 → 反思打分 → 放行 / 换参数重试 / 拒答。重试不回改写节点（改写 prompt 没变，重跑没意义），只换检索参数：`top_k` 放大 / 阈值下调 / 拆子问题。
- **引用白名单校验**：回答里的每条法条引用都必须在 `law_articles`（条文权威源表）里查得到。

**数据与可复现性**

- **按条切分 + 策略注册表**：`statute`（法律法规，按条切）/ `faq`（一问一块）/ `generic`（兜底），解析失败时按「结构感知 → 递归分隔符 → 定长」降级，**永不返回空结果**，并留痕实际用的策略与置信度。
- **26 字段元数据契约**：`chunk` 携带法名、条文号（原文 + 归一化）、章节路径、版本 ID、生效/失效日期、时效状态、地域、切分策略等，与 Milvus 标量列一一对应，有专门的契约测试防漂移。
- **版本对**：同一部法的多个版本共存（如《劳动法》2009 修正 + 2018 修正），旧版标 `amended` 默认被过滤；关掉过滤仍能召回，用来证明"过滤确实生效"而不是"数据压根没有"。
- **评测驱动**：52 题评测集 + 两份基线（direct / agentic）都是**签入版本库的资产**，指标按「法名 + 条文号」匹配，不用 LLM 当裁判。
- **删除即清理**：删除文档会同时删掉 Milvus 里的向量 —— `document_id` 曾被框架静默覆盖成 `"None"`，导致按 ID 删向量永远匹配不到，留下孤儿向量且不报错。这个坑修过，也有回归测试。
- **脱网测试**：全部单测不碰网络、不碰中间件，可在 CI 里直接跑。

---

## 三、当前实测数字

语料：9 份官方 docx = 7 部法 / **703 个条文片段**；评测集 **52 题**（36 道可判定 + 16 道超纲）。

**direct 基线**（单次混合检索，重排开、时效过滤开，`data/eval/baseline.json`，2026-10-20）

| Recall@5 | 完整召回@5 | MRR | NDCG@5 | 旧版混入率 | 未命中 |
|---|---|---|---|---|---|
| **0.9167** | 0.8056 | 0.7315 | 0.7612 | **0.0000** | 3 条 |

**agentic 基线**（生产路径：改写 → 召回 → 重排 → 反思 → 重试/拒答，`data/eval/baseline-agentic.json`，2026-10-21）

| 端到端 Recall@5 | 完整召回 | MRR | NDCG@5 | 放行子集 Recall@5 | 库内误拒 | 超纲拒答 |
|---|---|---|---|---|---|---|
| **0.8611** | 0.8056 | 0.7775 | 0.7605 | 0.9118 (n=34) | 2/36 | **16/16 = 1.0** |

> **两条路径必须分开看**：生产走 agentic，而评测默认走 direct。同一处修复可能在一条链上生效、另一条链上完全测不到 —— P1 那次"评测全绿、生产却把所有正常问题都拒答"就是从这道缝里漏过去的（详见 `docs/05-测试与评测.md` §八）。
> agentic 的端到端 recall **不能**与 direct 的数字相减：工作流一旦拒答就清空候选，这题在指标里记 0，于是"没检到"和"检到了但反思不放行"会混成一个数。

---

## 四、技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| Web 框架 | FastAPI + Uvicorn | REST + SSE 流式输出 |
| RAG 框架 | LlamaIndex | 文档加载 / 向量化 / Milvus 集成 |
| 工作流 | LangGraph | Agentic 检索的 9 节点状态机 |
| 生成模型 | DeepSeek（`deepseek-v4.1-flash`） | 改写、反思、生成统一用一个 |
| 向量模型 | 智谱 `embedding-3`（1024 维） | API 调用，无需本地下载 |
| 重排模型 | `BAAI/bge-reranker-base` | Cross-Encoder，**本地 CPU 推理**，float32 约 1GB 常驻 |
| 向量库 | Milvus 2.4（standalone，含 etcd + MinIO） | 26 个标量字段，支持查询表达式过滤 |
| 元数据/历史 | MySQL 8.0（5 张表） | `documents` / `law_versions` / `law_articles`（条文权威源）/ `conversations` / `messages` |
| 缓存 | Redis 7 | 查询改写与反思结果缓存（让评测可复现，同题两次不抖） |
| 前端 | Streamlit 单文件 | 上传、Top-K/阈值调节、流式问答、会话管理 |
| 测试 | pytest | 25 个测试文件 / 366 条用例，**脱网可跑** |

---

## 五、它是怎么跑起来的

### 5.1 入库链路

```
DOCX/PDF/MD/TXT 上传（或 index-dir 批量导入）
  ├─ 扩展名校验 → 落盘 data/raw/
  ├─ 内容 SHA-256 去重（重复上传返回 409，不重复入库）
  ├─ DocumentLoader 多格式加载
  ├─ resolve_ingestor：①用户指定 doc_type ②结构探测器 ③文件名规则 ④generic 兜底
  ├─ 按条切分 → ParsedSection[]（解析不出结构则降级，留痕 fallback_from）
  ├─ build_text_nodes：拼装 26 字段元数据
  │    ├─ 章节路径拼进 node.text（只用于向量化，提升召回）
  │    └─ display_text 另存干净原文（用于引用展示，两者刻意分开）
  ├─ 智谱 embedding → 写 Milvus（维度自动探测，建库前校验）
  ├─ 写 MySQL documents / law_articles（条文的权威源）
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
  ├─ 引用白名单校验（不存在的条文号一律标注"未能核验"）
  └─ 落库 messages（含 retrieval_info、tokens_used、duration_ms）
```

完整逐节点说明与设计态对照见 `docs/10-执行流程图.md`，表结构与字段语义见 `docs/13-数据模型与处理链路.md`。

---

## 六、快速开始

### 0. 前置要求

| 项 | 要求 |
|---|---|
| Python | 3.11+（开发环境实测 3.11 / 3.12） |
| Docker | 需要能跑 5 个容器（MySQL + Redis + Milvus + etcd + MinIO） |
| 内存 | 中间件 + 后端进程合计约 **3~4 GB**；重排器 float32 加载独占 ~1 GB |
| API Key | 智谱开放平台（供 embedding-3 向量）+ DeepSeek（供改写/反思/生成），两家都按用量计费，需各自准备额度 |
| 磁盘 | Milvus + MySQL 卷约需 5 GB |

> 重排器加载约需 30~35 秒，服务是**启动期预热**的（不是第一次请求时才加载），所以看到 `系统启动完成` 之前多等一会儿是正常的。

### 1. 克隆并安装依赖

```bash
git clone <你的仓库地址> legal-rag
cd legal-rag

# 建议用虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> `requirements.txt` 里几个依赖看着"多余"但**不能删**：`streamlit`（前端入口）、`jieba`（缺失时 BM25 会**静默降级**成字符 bigram，不报错、只是质量变差）、`torch`（`retriever.py` 直接按 dtype 分支 import 它）、`pytest`。每一条后面都写了理由。

### 2. 起中间件

```bash
docker compose up -d
docker compose ps        # 5 个容器都要 Up；milvus 依赖 etcd 与 minio，首次启动约 1~2 分钟
```

MySQL 首次启动会自动执行 `sql/init.sql` 建好 5 张表。**注意**：这个初始化只在卷为空时发生一次；如果之前起过 MySQL 又改了 `init.sql`，需要先 `docker compose down -v`（会清空数据卷）再 `up -d`。`sql/migrations/` 是给**已有数据库**升级用的，全新环境不用手动跑。

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填这两个
#   ZHIPU_API_KEY=...      # 智谱，供 embedding-3
#   DEEPSEEK_API_KEY=...   # DeepSeek，供改写/反思/生成
```

`.env` 已被 `.gitignore` 忽略，且 `tests/test_no_secrets.py` 会检查 `.env` 是否被 git 跟踪、跟踪文件与 git 历史里有没有密钥特征 —— **历史是永久的**，提交过一次就得重写历史才能清掉，所以这道检查放在提交前而不是事后。

### 4. 准备语料（重要）

`data/raw/` 被 `.gitignore` 忽略（体积大、可重新下载），**克隆下来是空的**，需要你自行获取法规原文：

- **来源**：国家法律法规数据库 `https://flk.npc.gov.cn/` —— 站内搜索 → 勾选条目 → 「批量下载文件」，选 `.docx`（官方排版正文）。
- **不要用 AI 生成的文本**。仓库里 `scripts/_legacy/prepare_data.py` 是历史遗留的生成脚本，已明确废弃：生成的文本与官方条文**号码一致但内容有出入**，且来源不可核验 —— 底本若是模型生成的，"每条依据都能点开核验"这个承诺从根上就不成立。
- **命名规则**：`<法名>_<公布或修正日期YYYYMMDD>.docx`，日期后缀**必须保留**。版本表（`law_versions`）按**文件名**匹配语料清单，文件名错了 → 时效状态拿不到 → 时效过滤不生效。
- 该库只收宪法/法律/行政法规/监察法规/地方性法规/司法解释，**不收部门规章**（如《工资支付暂行规定》要去人社部官网取）。
- 版权依据：《著作权法》第五条 —— 法律法规及具有立法、行政、司法性质的文件**不适用著作权法**。

当前清单（7 部法 / 9 个版本 / 703 条）记录在 `data/corpus_manifest.json`，含每份文件的 SHA-256：

```bash
python scripts/verify_corpus.py            # 校验缺失 / 多余 / 内容被改
python scripts/verify_corpus.py --update   # 你换了文件版本后，用当前文件刷新清单哈希
```

> 想跑通全部演示与评测基线，就按清单下同样的 9 份文件；只想试试流程，也可以只放一两部法，但**下面的指标数字就不成立了**。

### 5. 灌版本种子数据 → 启动服务 → 索引语料

```bash
# ① 把语料清单里的版本信息写进 law_versions（幂等，可反复执行）
python scripts/seed_law_versions.py --dry-run   # 先看会写什么
python scripts/seed_law_versions.py

# ② 启动后端（首次会加载重排器，约 30~35 秒）
python app/main.py
#   或开发模式（带热重载）： uvicorn app.main:app --reload --port 8000

# ③ 健康检查：确认 mysql / redis / milvus 三个都是 true
curl http://localhost:8000/api/health
```

另开一个终端，把 `data/raw/` 下的文档全部索引：

```bash
curl -X POST "http://localhost:8000/api/knowledge/index-dir?dir_path=data/raw&doc_type=statute&kb_id=default"

# 确认入库结果：documents 有行、law_articles 应有 703 行、Milvus 集合有向量
python scripts/probe_vector_store.py
```

最后起前端：

```bash
streamlit run streamlit_app.py    # 默认 http://localhost:8501
```

### 6. 问一句试试

在 Streamlit 里输入：

> 我在公司干了两年三个月，被辞退能拿多少补偿？

期望：回答给出「N 个月工资」的结论 + **《劳动合同法》第四十七条**（条文号正确）+ 行动建议，点开引用能看到逐字原文与"2012 年修正 · 现行有效"标注。再问一句超纲的（"火星上的房产税怎么交"），期望是**明确拒答且不出现任何法条编号**。

---

## 七、日常使用

### 7.1 Web UI（Streamlit）

侧边栏可调 Top-K、准入阈值、上传文档、查看/删除会话。界面上的上传只开放 `pdf/docx/md/txt` 四类，而后端 `POST /api/knowledge/upload` 实际还支持 `xlsx/csv` —— 批量导入请用 `index-dir` 或直接调 API。

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

交互式文档：服务起后访问 <http://localhost:8000/docs>。

一次带完整留痕的问答：

```bash
curl -s http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"公司三个月没发工资，我可以怎么做？","model_name":"deepseek-v4.1-flash","top_k":5,"use_agentic_rag":true}' \
  | python -m json.tool
```

响应里的 `sources[].law_name / article_no / validity_label / version_label / score_normalized` 就是每条依据的完整出处。`workflow_steps` 是一串**带参数的节点名**，能直接读出这次跑了什么：`query_analysis:cache`（改写命中缓存）、`law_filter(...)`、`domain_gate(0.4123->high)`、`rerank`、`reflection(0.86)`、`retry#1({...})`、`refuse(原因)`。

更细的复盘线索不在响应里，而在 **`messages.retrieval_info`（JSON 列）**：实际用于检索的改写句 `rewritten_query`、反思的结构化判断（`core_basis` / `coverage` / `missing`）、以及 `threshold_usage`（这次阈值到底比了什么、top1 多少、保留了几条）。排查"为什么答错了"就查这一列，或跑 `run_eval` 看逐题报告。

### 7.3 换语料的正确顺序（顺序不能反）

```bash
# 删除 data/raw 下的文件 **不会** 删掉 Milvus 里的向量，
# 旧片段会继续被召回且不报错。所以：
python scripts/verify_corpus.py            # ① 先看清单与文件是否一致
python scripts/reset_vector_store.py       # ② 清掉旧向量（支持 --dry-run）
# ③ 重新 seed_law_versions → 重新 index-dir → probe_vector_store.py
pytest                                     # ④ 全绿
python scripts/run_eval.py --check         # ⑤ 标准答案是否仍在语料里
python scripts/run_eval.py --top-k 5       # ⑥ 跑基线对比
```

> 语料一变，拒答闸门的两个阈值（`DOMAIN_SCORE_REJECT` / `DOMAIN_SCORE_CAUTION`）**必须重标定**，见下一节。

---

## 八、配置项速查

全部配置在 `app/config/settings.py`，从 `.env` 读取，`.env.example` 里每个键都有注释。下面只列**容易搞错语义**的几个：

| 键 | 默认 | 真实语义 |
|---|---|---|
| `SIMILARITY_THRESHOLD` | 0.1 | 名字叫"相似度阈值"，实际比的是**重排归一化分**（0~1，不是余弦相似度）；且只看本批**第 1 名**：top1 达标就保留整批，最终条数交给 `TOP_K`。它是"这批值不值得答"的粗闸门，**不是每条片段的及格线**。数值来自评测集标定，语料/切分一变就不可比 |
| `DOMAIN_SCORE_REJECT` / `DOMAIN_SCORE_CAUTION` | 0.345 / 0.36 | 拒答闸门，判据是**向量原始相似度最高值**（与重排分是两个量纲）。用 `python scripts/calibrate_domain_gate.py` 在评测集上跑出库内/超纲两分布后选点。当前两分布之间的**空档只有 0.0105** —— 换嵌入模型、扩语料、加超纲题都会让它静默失效（表现只是"超纲题又被回答了"） |
| `ENABLE_VALIDITY_FILTER` | true | 开：`validity_status == "effective"` 下推进 Milvus 查询表达式，旧版本进不了候选。关：旧版也能召回 —— 这是**对照组**，用来证明过滤真的生效，而不是数据本来就没有 |
| `ENABLE_AGENTIC_RAG` | true | 关掉后工作流**真的不执行那些节点**（服务端日志可见），不是"前端不传参"那种伪生效 |
| `RERANK_TORCH_DTYPE` | float32 | 内存吃紧时的第一档缓解手段（float16 / bfloat16 把权重内存砍半）。CPU 上 float16 部分算子支持有限，报错就退回 float32 或用 bfloat16 |
| `HF_HUB_OFFLINE` | true | 默认只读本地缓存。设为 false 才会去 huggingface.co 下载模型（首次装重排器时需要临时改） |
| `EMBEDDING_MODEL_NAME` | embedding-3 | **死配置**：没有任何代码读它。实际模型名在 `app/core/rag/embedder.py` 里硬编码。改它不会生效 |

---

## 九、测试与评测

### 9.1 单元测试（脱网，可在 CI 直接跑）

```bash
pytest                      # 25 个文件 / 366 条用例
pytest tests/test_metadata_contract.py -v
```

除了常规逻辑用例，还包含几条**元测试**（检查工程本身的地基）：

| 文件 | 防的是什么 |
|---|---|
| `test_dependencies.py` | requirements 漏声明（`jieba`、`pytest`、`streamlit` 都曾经漏过，本机 Anaconda 自带所以永不报错，换台机器直接 `command not found`） |
| `test_imports.py` | 模块路径断裂、循环依赖 |
| `test_no_secrets.py` | `.env` 被跟踪 / 密钥进历史 |
| `test_corpus.py` | 语料清单与文件不一致 |
| `test_schema_dictionary.py` | 表结构与 `docs/13` 的字段字典漂移（改了表没改文档 = 测试变红） |

### 9.2 检索评测

```bash
python scripts/run_eval.py --check                    # 只自检标准答案是否真在语料里，不依赖任何中间件
python scripts/run_eval.py --top-k 5                  # direct：纯检索，不调生成 LLM，无调用成本
python scripts/run_eval.py --top-k 5 --mode agentic   # agentic：经改写/重排/反思，会调 LLM（但不生成答案）
python scripts/run_eval.py --top-k 5 --mode gate      # 拒答闸门专项
python scripts/run_eval.py --top-k 5 --mode agentic --limit 3   # 冒烟，冒烟报告不作为基线
```

报告落 `data/eval/reports/run-<mode>-<时间戳>.json`，含**逐题明细**：召回了什么、每条的重排分、改写句、反思分、重试次数、阈值把哪几条砍了。指标口径与"为什么这么算"写在 `data/eval/README.md`。

评测集本身（`data/eval/legal_qa_eval.jsonl`）是**人工标注的标准答案，属于源代码级资产**，签入版本库；只有运行产物 `reports/` 被忽略。匹配按「法名 + 条文号」——只比条文号会把命中《劳动合同法》第三十六条判成命中《劳动法》第三十六条，**指标虚高且不报错**。

> **纪律**：不要为了让数字好看而回改标注。L21 改判后仍未命中，探针测出同一句口语问法下目标条文排候选池第 11、换成法言法语排第 1 —— 那是真实的检索缺口（口语→法律要素映射未实现），把尺子改弯去迁就检索等于把问题抹掉。

### 9.3 端到端验收

```bash
python scripts/e2e_acceptance.py    # 需服务已启动、语料已入库
```

人工验收 8 项清单见 `docs/05-测试与评测.md` §七。

---

## 十、项目结构

```
.
├── app/
│   ├── main.py                 # FastAPI 入口（lifespan 里做 BM25 重建 + 重排器预热）
│   ├── config/settings.py      # 全部配置与默认值，带"为什么这么定"的注释
│   ├── core/
│   │   ├── ingest/             # 文档类型注册表：statute / faq / generic + 元数据契约
│   │   ├── rag/                # loader / embedder / vector_store / retriever(混合检索+重排)
│   │   ├── workflow/           # LangGraph 9 节点 Agentic 工作流
│   │   ├── generate/           # 提示词组装 + 引用编号与白名单校验
│   │   ├── eval/               # 评测集加载、语料索引、指标（grade_hit / recall / MRR / NDCG）
│   │   ├── llm/                # 模型适配与路由
│   │   └── memory/             # MySQL 持久化 + Redis 缓存
│   ├── models/schemas.py       # 请求/响应模型（ChatResponse、SourceInfo 等）
│   ├── routers/                # chat / knowledge / conversation
│   ├── services/               # 编排层（rag_service 是入库与检索的主干）
│   └── utils/sse.py
├── scripts/                    # 运维与评测脚本（run_eval / verify_corpus / calibrate_domain_gate / ...）
│   └── _legacy/                # 已废弃脚本，保留只为留痕，不要运行
├── sql/
│   ├── init.sql                # 全新环境建表（5 张）
│   └── migrations/             # 已有库的升级迁移（001~004）
├── data/
│   ├── raw/                    # 语料 docx（gitignore，需自行下载官方文本）
│   ├── corpus_manifest.json    # 语料清单 + SHA-256，law_versions 的种子数据
│   └── eval/                   # 评测集、两份基线、README（指标口径说明）
├── tests/                      # 25 文件 / 366 条用例，脱网可跑
├── docs/                       # 需求 / 设计 / 任务清单 / 测试报告 / 流程图 / 数据模型
├── streamlit_app.py            # 前端单文件
├── docker-compose.yml          # MySQL + Redis + Milvus(+etcd+MinIO)
└── requirements.txt
```

---

## 十一、踩坑记录（FAQ）

**Q：启动卡在"加载重排器"好几分钟，是不是 OOM 了？**
多半是网络。sentence-transformers 加载**本地已缓存**的模型时，仍会先向 huggingface.co 发 HEAD 请求核对更新；国内网络下该请求超时，然后按 1s/2s/4s… 退避重试 5 次。默认 `HF_HUB_OFFLINE=true` 就是为了绕开它（代价：模型没缓存时直接报错而不是自动下载，首次下载时临时设为 `false`）。

**Q：OOM / 内存不够？**
`bge-reranker-base` 以 float32 加载约 1 GB 常驻，实测在开发机上 OOM 过。把 `RERANK_TORCH_DTYPE` 改成 `float16` 或 `bfloat16`，权重内存直接砍半。**重排器加载失败不影响正确性**（排序降一档），但会影响拒答 —— 所以拒答闸门刻意用了不依赖重排器的判据。

**Q：`MilvusException: the dim of field data is not equal to schema dim`？**
向量维度不一致。智谱 `embedding-3` 实际返回 **1024** 维，而早期代码硬编码 2048，这个不一致只在真正插入时才暴露 —— 那时已经白跑了几百次嵌入调用。现在 `EMBEDDING_DIM=0` 表示自动探测 + 建库前校验；只有在需要跨环境锁死维度时才显式赋值。

**Q：删了 `data/raw` 下的文件，旧内容还能被检索到？**
正常现象，也是刻意的坑位记录：删文件 ≠ 删向量。要走 [7.3 的顺序](#73-换语料的正确顺序顺序不能反)：`reset_vector_store.py` → 重新索引。删除**单个文档**用 `DELETE /api/knowledge/documents/{id}`，它会同步清理向量。

**Q：改了 `.env` 里的 `EMBEDDING_MODEL_NAME` 没生效？**
它本来就是死配置，见[配置速查](#八配置项速查)。

**Q：BM25 关键词一路似乎没用？**
检查 `jieba` 装上了没。`retriever.py` 在 jieba 缺失时会**静默降级**成字符 bigram —— 不报错，只是质量变差。

**Q：服务起了但 `/api/health` 显示 degraded？**
按返回体里 `mysql` / `redis` / `milvus` 哪个是 false 逐个查容器。Milvus standalone 依赖 etcd 与 MinIO，三个都得是 `Up`；`docker compose ps` + `docker compose logs milvus` 是最快的入口。注意 MySQL 宿主机端口是 **3307**，而 `.env` 里 `MYSQL_PORT` 必须跟着填 3307。

**Q：回答里没有引用，或引用被标成"未能核验"？**
先看 `sources[].article_no` 与 `law_articles` 表里是否一致（引用白名单查的是条文权威源）。若语料换过但没重建索引，或文件名不符合 `<法名>_<日期>.docx` 规则导致 `version_id` 拿不到，都会出现这种情况。`docs/13-数据模型与处理链路.md` 有一条完整链路走查可以对照排查。

---

## 十二、已知限制

写下来是为了不让人拿这些当"隐藏 bug"，也是为了标出下一步：

1. **时效管理只到"整部法"级，没有"条文级"失效。** 《劳动法》2018 修正版整部仍现行有效，所以它的**第八十二条（仲裁时效 60 日）仍会被召回** —— 尽管该条已被《劳动争议调解仲裁法》第二十七条（1 年）取代。文件级过滤抓不到这种"条文被新法取代"，这是当前最硬的一处缺口（待办 P2-05）。
2. **拒答闸门用的是绝对余弦阈值，两分布之间空档只有 0.0105。** 换嵌入模型、扩语料、加超纲题都会让它失效，且**失效是静默的**。计划改用相对判据（top1 与 top2 差值 / 分布标准化）。
3. **口语化问法召回差于法言法语。** 分类别看（2026-10-21 agentic 基线）：`term` 18 题端到端 **0.9444**，`colloquial` 14 题 **0.8571（12/14）**；重排**关闭**的旧基线下这两个数是 1.000 与 0.714 —— 差距几乎全压在"口语 → 法律术语"这一处（注意旧口径不能和开重排的数字相减）。铁证是 L21：问「公司一直不给我交社保，我该怎么办？」时《社会保险法》第六十三条排候选池第 11，换成法言法语「用人单位不缴纳社会保险费 责令限期缴纳 补缴」后同一条排第 1。需求 F2.7「口语化 → 法律要素映射」尚未实现（现有改写节点只做通用改写）。
4. **没有集成测试层。** 需要真实中间件的那条全链路仍靠手跑 `scripts/e2e_acceptance.py`；离线那层（契约测试、依赖审计、语料校验、密钥防线）已补齐。
5. **FAQ 文档类型没有验证语料**，`faq` 解析器有实现和单测，但端到端效果未被真实数据验证过。
6. **配置项仍在 `.env`（服务端生效、可验证），没有后台管理界面。** 落库 + 变更留痕是 P3 计划（`01-需求文档.md` F6）。
7. **`messages.retrieval_info` 之前为空的那批历史行补不回** —— 改动前的对话，当时的召回结果已无处可查。
8. **不做 OCR。** 扫描件 PDF 会被明确拒绝并说明原因，不做静默降级（个人自用的法规资料基本都是文字版；OCR 的识别错误会污染引用溯源）。
9. **单用户、单机**。没有多租户、登录、RBAC、配额 —— 曾经按"企业级"设计过，2026-10-13 明确收缩掉了，理由见 `docs/01-需求文档.md` §八 变更记录。

---

## 十三、文档索引

`docs/` 不是一次性说明，而是**开发过程的留痕**（需求 → 设计 → 任务清单 → 验收证据，每条都能指到 commit）。想改代码前建议先读 `09` 和 `13`。

| 文档 | 作用 |
|---|---|
| `01-需求文档.md` | 需求与**验收标准**（没有验收标准的不算需求）、明确不做的事、范围变更记录 |
| `02-设计文档.md` | 目标态设计，§13 是"设计 vs 现状"逐条对照表 |
| `02a-文档解析与切分策略设计.md` | 类型路由与切分策略设计 |
| `03-开发流程规范.md` | 流程纪律：bug 登记、变更留痕、验收要求 |
| `04-P0/P1/P2-任务清单.md` | 分批任务与验收标准，含完成记录与实测数字 |
| `05-测试与评测.md` | 测试分层、按功能点的用例、评测集与指标口径 |
| `06-P0测试执行报告.md` / `07-P0端到端验收与重排收益.md` | P0 收口的执行证据与重排收益对照 |
| `08-语料清单.md` | 每份语料的来源、版本、时效、校验和 |
| **`09-当前状态对账.md`** | **"此刻代码里到底有什么"** —— 需求条目逐条对账 + 已知缺陷清单。入门先读这篇 |
| `10-执行流程图.md` | 代码实际怎么跑（启动 / 入库 / 问答三条链路），与设计态对照 |
| `11-文档处理管线.md` | 解析与切分细节 |
| `13-数据模型与处理链路.md` | 表结构、字段含义、链路走查、变更登记表（**改表结构必须同步这里的字段字典，否则 `test_schema_dictionary` 变红**） |

---

## 十四、贡献与许可

**想提 PR 的话**：请先读 `docs/03-开发流程规范.md` 和 `docs/09-当前状态对账.md` §六（已知缺陷清单）。这个项目的纪律是——

- 改检索/切分参数后**必须跑评测回归**，并与对应基线比较（`direct` 的改动看 `data/eval/baseline.json`，工作流的改动看 `baseline-agentic.json`，两者数字**不可互减**）。
- 改表结构要同步 `sql/init.sql`、`sql/migrations/`、`docs/13` 的字段字典三处。
- 指标退步时，先证明是"尺子变了"还是"真的变差了"，再决定要不要回改标注。
- 语料变动后：`verify_corpus.py` → `reset_vector_store.py` → 重新索引 → 重标定拒答阈值。

**许可**：**待定** —— 仓库目前还没有 `LICENSE` 文件（初步倾向 MIT，但由作者最终确定）。正式开源前需要一并确认两件事：语料的许可边界（法规原文本身不适用著作权法，见《著作权法》第五条），以及第三方模型（`bge-reranker-base`、智谱 embedding API）的使用条款。

---

如果这个项目对你理解「怎么把 RAG 做到可验证」有一点参考价值，欢迎点个 Star。
