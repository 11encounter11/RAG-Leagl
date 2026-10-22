# `scripts/` 使用说明

这 6 个脚本是**你在命令行操作这个项目的全部入口**：核对语料、灌版本数据、重灌向量、查库、跑评测、重标阈值。后端本身不需要它们参与请求处理（`app/` 不 import 这里的任何文件），所以漏跑某一步不会报错，只会让某个能力静默失效——下表最后一列专门写这件事。

| 脚本 | 什么时候要跑 | 前置依赖 | 花钱吗 | 漏掉的后果 |
|---|---|---|---|---|
| `verify_corpus.py` | 每次下载/替换 `data/raw/` 里的文件之后 | 无（只读本地文件） | 否 | 语料下错版本没人报警，而检索质量、评测指标、时效判断全建在这批文件上 |
| `seed_law_versions.py` | 全新环境**第一次入库之前**，以及语料清单变动后 | MySQL | 否 | `law_versions` 是空表 → chunk 拿不到 `version_id` → 时效过滤与"现行有效"标注**全部失效**，但服务照常启动、不报错 |
| `reset_vector_store.py` | 换语料、换切分参数、换嵌入模型之前 | Milvus | 否 | 旧向量继续被召回（删 `data/raw/` 下的文件**不会**删向量） |
| `probe_vector_store.py` | 任何时候想确认"向量到底进没进、元数据对不对" | Milvus | 少量（读操作） | 无，纯诊断 |
| `run_eval.py` | 改了检索参数、切分策略、阈值、过滤逻辑之后 | 见下方三种模式 | 视模式而定 | 调参变成凭感觉，改坏了也没人告诉你 |
| `calibrate_domain_gate.py` | **语料变动 / 换嵌入模型之后**（这两件事一定会移动分数分布） | Milvus + 智谱嵌入 | 是：每道评测题一次查询嵌入 | 拒答闸门阈值失准，表现为"超纲题又被回答了"，而且不报错 |

## 逐个说明

### `verify_corpus.py` —— 语料核对

```bash
python scripts/verify_corpus.py            # 核对 data/raw 与 data/corpus_manifest.json
python scripts/verify_corpus.py --update   # 你主动换过文件后，用当前文件刷新清单里的哈希
```

检查三件事：清单里列的文件是否都在（缺失报错）、`data/raw` 里有没有多余文件（可能是下错了版本）、每份文件的 SHA-256 是否吻合（被改动过就报错）。

### `seed_law_versions.py` —— 灌版本表种子

```bash
python scripts/seed_law_versions.py --dry-run   # 只看会写什么，不落库
python scripts/seed_law_versions.py
```

读 `data/corpus_manifest.json`，把「哪部法有哪几个版本、各自什么时候公布与施行、时效状态是什么」写进 MySQL 的 `law_versions`。版本 ID 由（法名, 版本标签, 公布日期）确定性推导，所以**可以反复执行**，已存在的行会被更新，不产生重复。

必须在入库前跑：后端入库时按**文件名**去这张表取版本信息，取不到就退回占位值，时效过滤等于没开。

### `reset_vector_store.py` —— 清空向量集合

```bash
python scripts/reset_vector_store.py --dry-run   # 只统计，不删除
python scripts/reset_vector_store.py --yes       # 跳过交互确认
python scripts/reset_vector_store.py --collection knowledge_base   # 默认取 .env 里的配置
```

**不可逆**：集合里所有向量都会没，只能重新入库找回。所以先 `--dry-run`。

### `probe_vector_store.py` —— 只读体检

```bash
python scripts/probe_vector_store.py
```

打印集合条数、标量字段、若干条样本的元数据。第一次入库后跑它，能立刻看出"条文号有没有抽出来""章节路径对不对"。

### `run_eval.py` —— 检索质量评测

```bash
# ① 自检：标准答案是否真的存在于语料里。不连任何中间件、不调任何 API，改完评测集就该跑
python scripts/run_eval.py --check

# ② direct：单次混合检索（向量 + BM25 + RRF + 重排），产出 Recall@k / MRR / NDCG@k
python scripts/run_eval.py --top-k 5

# ③ agentic：生产走的完整工作流（改写 → 召回 → 重排 → 反思 → 重试/拒答）
python scripts/run_eval.py --top-k 5 --mode agentic
python scripts/run_eval.py --top-k 5 --mode agentic --limit 3   # 冒烟，冒烟报告不能当基线

# ④ gate：专测拒答闸门（超纲题有没有被拦住、库内题有没有被误杀）
python scripts/run_eval.py --top-k 5 --mode gate
```

| 模式 | 需要 | 会调用什么 | 大致耗时 |
|---|---|---|---|
| `--check` | 无 | 无 | 秒级 |
| `direct` | MySQL + Milvus + 重排器本地模型 | 每题一次查询嵌入 | 52 题几分钟 |
| `agentic` | 同上 + DeepSeek | 嵌入 + 改写/反思若干次 LLM 调用（**不生成答案**） | 15 分钟量级 |

报告写到 `data/eval/reports/run-<mode>-<时间戳>.json`，含逐题明细（召回了什么、每条重排分、改写句、反思分、阈值砍掉哪几条）。判定口径见 `data/eval/README.md`——**`direct` 与 `agentic` 两份基线的数字不能互相减**。

### `calibrate_domain_gate.py` —— 重标拒答阈值

```bash
python scripts/calibrate_domain_gate.py          # 打印库内/超纲两组分布
python scripts/calibrate_domain_gate.py --csv    # 另存逐题明细 CSV
```

把 52 道评测题分成"库内能答"和"超纲该拒"两组，各自跑一遍向量检索，打印两组最高相似度的分布，再据此选 `DOMAIN_SCORE_REJECT` / `DOMAIN_SCORE_CAUTION`。

**什么时候必须重跑**：换嵌入模型、语料增删、评测集加题。当前两分布之间的空档只有 0.0105，余量极窄，而阈值失效是静默的——不会报错，只会让超纲题重新被回答。
