-- ============================================================
-- 迁移 001: documents 表补齐 P0 字段
-- ============================================================
-- 背景(测试期发现, 记为 N-7):
--   SQLAlchemy 的 Base.metadata.create_all() 只会"建不存在的表",
--   **不会修改已存在的表**。所以当 models 里给 Document 加了字段后,
--   老库里的 documents 表仍是旧结构, 而代码已经开始查询新字段 ——
--   表现为运行期报 Unknown column, 而不是建表时报错。
--
--   这类问题在"本地已经跑过一版、代码又改了模型"的场景里几乎必然出现,
--   而且报错发生在查询语句里, 很难一眼看出是 schema 没跟上。
--
-- 用法:
--   docker exec -i legal-rag-mysql sh -c \
--     'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" legal_rag' < sql/migrations/001_documents_p0_columns.sql
--
-- 幂等性说明:
--   MySQL 8.0 不支持 ALTER TABLE ... ADD COLUMN IF NOT EXISTS,
--   因此本脚本**只能执行一次**。重复执行会报 Duplicate column name,
--   这反而是安全的 —— 报错说明已经迁移过了, 不会破坏数据。
--
-- P0 结论: 项目当前没有迁移工具, 靠手工 SQL 脚本 + 版本号管理。
--          这是明确的欠债, 列入 P1: 引入 Alembic。
-- ============================================================

ALTER TABLE documents
    ADD COLUMN original_filename VARCHAR(255) DEFAULT NULL COMMENT '上传时的原始文件名' AFTER filename,
    ADD COLUMN tenant_id VARCHAR(64) NOT NULL DEFAULT 'platform' COMMENT '租户ID' AFTER error_message,
    ADD COLUMN kb_id VARCHAR(64) NOT NULL DEFAULT 'public' COMMENT '知识库ID' AFTER tenant_id,
    ADD COLUMN content_hash VARCHAR(64) DEFAULT NULL COMMENT '文件内容SHA256，用于重复上传去重' AFTER kb_id,
    ADD COLUMN doc_type VARCHAR(32) DEFAULT NULL COMMENT '文档类型：statute/policy/faq/generic' AFTER content_hash,
    ADD COLUMN doc_type_source VARCHAR(32) DEFAULT NULL COMMENT '类型判定来源：user/filename/detector/generic' AFTER doc_type,
    ADD COLUMN chunk_strategy VARCHAR(64) DEFAULT NULL COMMENT '实际使用的切分策略' AFTER doc_type_source,
    ADD COLUMN strategy_confidence VARCHAR(16) DEFAULT NULL COMMENT '策略置信度' AFTER chunk_strategy,
    ADD COLUMN structure_detected VARCHAR(128) DEFAULT NULL COMMENT '结构探测器识别结果' AFTER strategy_confidence;

ALTER TABLE documents
    ADD INDEX idx_tenant_kb (tenant_id, kb_id),
    ADD INDEX idx_content_hash (content_hash);
