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
-- tenant_id / kb_id 从第一张表就带上, 避免 P2 做多租户时全库迁移。
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(64) PRIMARY KEY COMMENT '文档ID（UUID）',
    filename VARCHAR(255) NOT NULL COMMENT '落盘的文件名',
    original_filename VARCHAR(255) DEFAULT NULL COMMENT '上传时的原始文件名',
    file_type VARCHAR(20) NOT NULL COMMENT '文件类型：pdf/docx/xlsx/csv/md/txt',
    file_size BIGINT DEFAULT 0 COMMENT '文件大小（字节）',
    chunk_count INT DEFAULT 0 COMMENT '分块数量',
    status VARCHAR(20) DEFAULT 'pending' COMMENT '状态：pending/processing/completed/failed',
    error_message TEXT DEFAULT NULL COMMENT '错误信息',

    tenant_id VARCHAR(64) NOT NULL DEFAULT 'platform' COMMENT '租户ID',
    kb_id VARCHAR(64) NOT NULL DEFAULT 'public' COMMENT '知识库ID',

    content_hash VARCHAR(64) DEFAULT NULL COMMENT '文件内容SHA256，用于重复上传去重',
    doc_type VARCHAR(32) DEFAULT NULL COMMENT '文档类型：statute/policy/faq/generic',
    doc_type_source VARCHAR(32) DEFAULT NULL COMMENT '类型判定来源：user/filename/detector/generic',
    chunk_strategy VARCHAR(64) DEFAULT NULL COMMENT '实际使用的切分策略',
    strategy_confidence VARCHAR(16) DEFAULT NULL COMMENT '策略置信度',
    structure_detected VARCHAR(128) DEFAULT NULL COMMENT '结构探测器识别结果',

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间',
    processed_at DATETIME DEFAULT NULL COMMENT '处理完成时间',
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_tenant_kb (tenant_id, kb_id),
    INDEX idx_content_hash (content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识库文档表';
