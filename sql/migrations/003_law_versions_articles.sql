-- ============================================================
-- 迁移 003: 新增 law_versions / law_articles 两张表（P1-1 / T-23）
-- ============================================================
-- 背景:
--   在此之前, 条文的原文与版本/时效元数据**只存在向量库里**。向量库本该是
--   "可重建的派生数据", 却兼任了"原文唯一副本" —— 结果两件事都做不好:
--     · Milvus 数据丢了, 版本与时效元数据无从恢复
--     · 引用白名单校验只能查 Milvus, 而那里可能已被污染
--   这两张表建完之后: MySQL 是权威源, Milvus 退回索引角色。
--
-- 作用范围:
--   **只建表, 不灌数据**。灌版本种子是 T-24, 入库时写条文是 T-25。
--
-- 幂等性:
--   与 001 不同 —— 本脚本用的是 CREATE TABLE IF NOT EXISTS, **可重复执行**。
--   (001 只能跑一次, 因为它用 ALTER TABLE ADD COLUMN, MySQL 8.0 不支持 IF NOT EXISTS)
--
-- 用法:
--   docker exec -i legal-rag-mysql sh -c \
--     'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" --default-character-set=utf8mb4 legal_rag' \
--     < sql/migrations/003_law_versions_articles.sql
--
-- 注意:
--   全新环境无需执行本脚本 —— sql/init.sql 已包含这两张表,
--   且应用启动时 Base.metadata.create_all() 也会自动建缺失的表。
--   本脚本是为"已有数据的旧库"和"不跑应用的初始化路径"准备的。
--
-- 字段严格按 docs/02-设计文档.md §3.1, 不加不减。
-- 索引名与 SQLAlchemy 自动生成的保持一致, 避免两种初始化路径产生不同结构。
-- ============================================================

CREATE TABLE IF NOT EXISTS law_versions (
    id VARCHAR(64) PRIMARY KEY COMMENT '版本ID',
    document_id VARCHAR(64) NOT NULL COMMENT '来源文档ID（→documents.id）',
    law_name VARCHAR(200) NOT NULL COMMENT '法名，同一部法的多个版本靠它归组',
    law_level ENUM('宪法','法律','行政法规','地方性法规','司法解释','部门规章','其他')
        DEFAULT NULL COMMENT '法律层级',
    issuing_body VARCHAR(200) DEFAULT NULL COMMENT '制定机关',
    doc_no VARCHAR(100) DEFAULT NULL COMMENT '发文字号',
    version_label VARCHAR(100) DEFAULT NULL COMMENT '版本标签，如 2007年通过 / 2012年修正',
    publish_date DATE DEFAULT NULL COMMENT '公布日期',
    effective_date DATE DEFAULT NULL COMMENT '施行日期（判断历史法条适用哪一版看它）',
    expiry_date DATE DEFAULT NULL COMMENT '失效日期（NULL = 现行有效）',
    validity_status ENUM('effective','amended','repealed','pending')
        NOT NULL DEFAULT 'effective' COMMENT '时效状态',
    superseded_by VARCHAR(64) DEFAULT NULL COMMENT '被哪个版本取代（→law_versions.id）',
    source_url VARCHAR(500) DEFAULT NULL COMMENT '来源链接',
    INDEX ix_law_versions_document_id (document_id),
    INDEX ix_law_versions_law_name (law_name),
    INDEX ix_law_versions_validity_status (validity_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='法规版本表';

CREATE TABLE IF NOT EXISTS law_articles (
    id VARCHAR(64) PRIMARY KEY COMMENT '条文ID',
    version_id VARCHAR(64) NOT NULL COMMENT '所属版本（→law_versions.id）',
    article_no VARCHAR(50) NOT NULL COMMENT '条文号，如 第四十七条',
    article_no_norm INT DEFAULT NULL COMMENT '归一化条文号，如 47（用于排序/匹配）',
    chapter_path VARCHAR(500) DEFAULT NULL COMMENT '章节路径',
    seq INT DEFAULT NULL COMMENT '在版本内的顺序号',
    content TEXT NOT NULL COMMENT '条文原文',
    INDEX ix_law_articles_version_id (version_id),
    INDEX idx_version_article (version_id, article_no_norm)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='条文表（权威源）';
