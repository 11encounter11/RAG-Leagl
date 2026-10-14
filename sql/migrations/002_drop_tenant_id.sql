-- ============================================================
-- 002 v0.2/v0.3 范围收缩对齐: 移除 tenant_id + 清理 doc_type 注释
--        (多租户 -> 个人助手; 内部制度文档类型 -> 已取消)
-- ============================================================
-- 背景:
--   需求文档 v0.1 曾计划做多租户, 于是 documents 表从第一张表就带上了
--   tenant_id 作为占位字段。v0.2 决定只做个人自用助手, 多租户不做了。
--
--   这个占位字段从未承载任何真实逻辑(全程固定值 'platform'), 留着只会
--   让人以为系统支持多租户 —— 属于"半砍"状态, 比全删更糟。
--   故新增本迁移显式删除。
--
--   同理, 001 给 doc_type 写的注释枚举了 policy(单位内部制度), 该类型已随
--   v0.3 删除(PolicyIngestor 已从代码移除), 注释里的枚举就是当前唯一的
--   政策类文档入口, 故一并纠正为 statute/faq/generic。
--
-- 注意:
--   init.sql 已同步为不含 tenant_id 的新版; 全新环境无需执行本脚本。
--   本脚本只用于把"已经跑过 001 的旧库"迁移到新 schema。
--
-- 字段命名: 原索引 idx_tenant_kb 同时覆盖 (tenant_id, kb_id),
--           删除 tenant_id 后需要重建为只覆盖 kb_id 的 idx_kb。
-- ============================================================

ALTER TABLE documents
    DROP INDEX idx_tenant_kb;

ALTER TABLE documents
    DROP COLUMN tenant_id;

ALTER TABLE documents
    ADD INDEX idx_kb (kb_id);

-- doc_type 枚举注释同步(仅改注释, 类型与默认值不变, 不动数据)
ALTER TABLE documents
    MODIFY COLUMN doc_type VARCHAR(32) DEFAULT NULL COMMENT '文档类型：statute/faq/generic';
