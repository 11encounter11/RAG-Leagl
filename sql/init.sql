-- 法律智能问答系统 - 数据库初始化脚本

-- 会话表
CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR(64) PRIMARY KEY COMMENT '会话ID（UUID）',
    title VARCHAR(200) DEFAULT '新对话' COMMENT '会话标题',
    model_name VARCHAR(50) NOT NULL COMMENT '使用的模型名称',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_created_at (created_at),
    INDEX idx_model_name (model_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='会话表';

-- 消息表
CREATE TABLE IF NOT EXISTS messages (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '消息ID',
    conversation_id VARCHAR(64) NOT NULL COMMENT '所属会话ID',
    role VARCHAR(20) NOT NULL COMMENT '角色：user/assistant/system',
    content TEXT NOT NULL COMMENT '消息内容',
    model_name VARCHAR(50) DEFAULT NULL COMMENT '使用的模型（assistant消息）',
    tokens_used INT DEFAULT 0 COMMENT '消耗的 token 数',
    retrieval_info JSON DEFAULT NULL COMMENT '检索信息（命中的文档片段等）',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_conversation_id (conversation_id),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='消息表';

-- 知识库文档表
-- 与 app/core/memory/db_store.py 的 Document 模型必须保持一致。
-- kb_id 作为知识库分组(如按法律领域/主题分库)。
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(64) PRIMARY KEY COMMENT '文档ID（UUID）',
    filename VARCHAR(255) NOT NULL COMMENT '落盘的文件名',
    original_filename VARCHAR(255) DEFAULT NULL COMMENT '上传时的原始文件名',
    file_type VARCHAR(20) NOT NULL COMMENT '文件类型：pdf/docx/xlsx/csv/md/txt',
    file_size BIGINT DEFAULT 0 COMMENT '文件大小（字节）',
    chunk_count INT DEFAULT 0 COMMENT '分块数量',
    status VARCHAR(20) DEFAULT 'pending' COMMENT '状态：pending/processing/completed/failed',
    error_message TEXT DEFAULT NULL COMMENT '错误信息',

    kb_id VARCHAR(64) NOT NULL DEFAULT 'default' COMMENT '知识库ID',

    content_hash VARCHAR(64) DEFAULT NULL COMMENT '文件内容SHA256，用于重复上传去重',
    doc_type VARCHAR(32) DEFAULT NULL COMMENT '文档类型：statute/faq/generic',
    doc_type_source VARCHAR(32) DEFAULT NULL COMMENT '类型判定来源：user/filename/detector/generic',
    chunk_strategy VARCHAR(64) DEFAULT NULL COMMENT '实际使用的切分策略',
    strategy_confidence VARCHAR(16) DEFAULT NULL COMMENT '策略置信度',
    structure_detected VARCHAR(128) DEFAULT NULL COMMENT '结构探测器识别结果',

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间',
    processed_at DATETIME DEFAULT NULL COMMENT '处理完成时间',
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_kb (kb_id),
    INDEX idx_content_hash (content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识库文档表';

-- 法规版本表（P1）
-- 与 app/core/memory/db_store.py 的 LawVersion 模型必须保持一致。
-- 同一部法的多个版本靠 law_name 归组；validity_status 决定检索时是否召回该版本。
--
-- 索引名刻意与 SQLAlchemy 自动生成的一致（ix_<表>_<列>），
-- 否则"用 init.sql 初始化的库"与"用 create_all 建的库"索引名会不同，
-- 以后写迁移脚本时容易对不上。
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

-- 条文表（P1）—— **权威源（ground truth）**
-- 与 app/core/memory/db_store.py 的 LawArticle 模型必须保持一致。
-- 向量库里的 chunk 是它的检索索引，不是权威副本：引用白名单校验查本表。
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
