-- ============================================================
-- 迁移 004: messages 表新增耗时两列（P2-03，需求 F8.2 剩余部分）
-- ============================================================
-- 背景:
--   需求 §五「可观测性」写的是: 每次请求可追溯到「召回了什么、用了什么策略、耗时分布」。
--   P1 的 T-31 已把前两项落库(retrieval_info), 但**耗时没有** —— response_time
--   只在接口响应里返回, 库里没有对应列。结果 §6.4 那条 P95 < 3s 的目标一直无法验证,
--   而 P2-08「长对话压缩摘要」的验收(20 轮后上下文/耗时不线性增长)也缺数据源。
--
-- 为什么允许 NULL 而不是 DEFAULT 0:
--   非流式问答没有"首 token"这个概念。存 0 会被读成"快得可以忽略",
--   而真实含义是"不适用"。这与 T-31 里"适配器没回报用量时返回 None 而不是伪造 0"
--   是同一条原则 —— 别把"没测"写成"是零"。
--
-- 幂等性:
--   与 003 不同, 本脚本**只能执行一次**。MySQL 8.0 不支持 ADD COLUMN IF NOT EXISTS,
--   重复执行会报 "Duplicate column name"。(001 同理, 见其文件头说明)
--   执行前如需确认, 先跑: SHOW COLUMNS FROM messages LIKE 'duration_ms';
--
-- 为什么必须手工跑这一趟:
--   应用启动时 Base.metadata.create_all() 只会建**缺失的表**, 不会给已有表加列。
--   全新环境无需执行本脚本(建表时直接带上这两列)。
--
-- 用法:
--   docker exec -i legal-rag-mysql sh -c \
--     'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" --default-character-set=utf8mb4 legal_rag' \
--     < sql/migrations/004_message_timing.sql
-- ============================================================

ALTER TABLE messages
    ADD COLUMN duration_ms    INT NULL COMMENT '整次请求耗时(毫秒): 收到请求 -> 落库/流结束',
    ADD COLUMN first_token_ms INT NULL COMMENT '首 token 耗时(毫秒, 仅流式; NULL = 不适用)';
