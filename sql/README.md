# `sql/` 说明

只有一个文件：`init.sql`，建 **5 张表**。

## 它什么时候被执行

`docker-compose.yml` 把它挂进 MySQL 容器的初始化目录：

```yaml
- ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
```

所以**全新环境不用手动跑**——第一次 `docker compose up -d` 时自动建好。两个注意点：

1. **只在数据卷为空时执行一次**。如果你之前起过 MySQL 又改了 `init.sql`，得先 `docker compose down -v`（清空数据卷）再 `up -d`，否则新列不会凭空出现。
2. 脚本第一行是 `SET NAMES utf8mb4;`，别删。MySQL 容器首次执行时客户端默认字符集是 latin1，会把 `law_level`、`validity_status` 这些**中文 ENUM 值**按 latin1 写进 utf8mb4 表，表现为 `Data truncated`。

不用 Docker 时手动初始化：

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS legal_rag DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;"
mysql -u root -p legal_rag < sql/init.sql
```

## 5 张表谁在写、谁在读

| 表 | 内容 | 写入时机 | 谁读它 |
|---|---|---|---|
| `conversations` | 会话（标题、模型、时间） | 第一次提问时自动创建 | 会话列表接口 |
| `messages` | 每条消息，含 `retrieval_info`(JSON 留痕)、`tokens_used`、`duration_ms`、`first_token_ms` | 每次问答 | 历史消息接口、事后复盘"这次为什么答错了" |
| `documents` | 已入库文档：原始文件名、内容哈希（去重）、`doc_type`、实际切分策略与置信度、块数 | 上传/批量索引 | 文档列表接口 |
| **`law_versions`** | 法规版本层：法名、版本标签、公布/施行日期、`validity_status` | **`scripts/seed_law_versions.py`**（不跑就是空表，时效过滤静默失效） | 入库时按文件名取 `version_id`；检索时把 `effective` 下推进 Milvus 过滤表达式 |
| **`law_articles`** | 条文层，**权威源**：条文号（原文 + 归一化）、章节路径、条文全文 | 每次入库自动写 | 引用白名单校验查这张表——查不到对应行就在回答上标"部分依据未能核验" |

一句关键设计：**向量库里的 chunk 只是 `law_articles` 的检索索引，不是权威副本。** 所以"引用是否真实存在"必须查 MySQL 的条文表，而不是拿召回结果自证。

## 为什么这里没有迁移脚本

发布线只提供**全新建库**这一条路径。`init.sql` 就是当前的完整 schema。如果你手上跑着更早期版本建出来的库、列对不上，最省事的做法是导出你的对话数据、清空重建、重新灌语料（条文与向量都能从语料重生成，只有历史对话不可再生）。
