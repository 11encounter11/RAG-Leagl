"""
MySQL 数据库存储 - 对话历史持久化
使用 SQLAlchemy ORM
"""
import uuid
import logging
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    Integer,
    DateTime,
    Date,
    Enum,
    Index,
    JSON,
    ForeignKey,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

from app.config.settings import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


# ===== 数据库模型 =====
class Conversation(Base):
    """会话表"""
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True, comment="会话ID")
    title = Column(String(200), default="新对话", comment="会话标题")
    model_name = Column(String(50), nullable=False, comment="使用的模型")
    created_at = Column(DateTime, default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """消息表"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="消息ID")
    conversation_id = Column(String(64), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False, comment="角色")
    content = Column(Text, nullable=False, comment="消息内容")
    model_name = Column(String(50), nullable=True, comment="模型名称")
    tokens_used = Column(Integer, default=0, comment="消耗token数")
    retrieval_info = Column(JSON, nullable=True, comment="检索信息")
    # 耗时(P2-03)。**刻意不用 default=0**: 非流式没有"首 token"概念, 0 会被读成
    # "快到可以忽略", 而真值是"没这个量" —— 与 tokens_used 的"0 表示确实没调用"区分开。
    duration_ms = Column(Integer, nullable=True, comment="整次请求耗时(毫秒)")
    first_token_ms = Column(Integer, nullable=True, comment="首 token 耗时(毫秒, 仅流式)")
    created_at = Column(DateTime, default=func.now(), comment="创建时间")

    conversation = relationship("Conversation", back_populates="messages")


class Document(Base):
    """知识库文档表。

    P0 新增字段都是为了回应两个具体验收:
      - 追溯   : original_filename / content_hash / doc_type / chunk_strategy 必须落库,
                 否则"这条片段是怎么切出来的"无从审计;
      - 分组   : kb_id 作为知识库分组(如按法律领域/主题分库)。
    """
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, comment="文档ID")
    filename = Column(String(255), nullable=False, comment="文件名")
    original_filename = Column(String(255), nullable=True, comment="上传时的原始文件名")
    file_type = Column(String(20), nullable=False, comment="文件类型")
    file_size = Column(Integer, default=0, comment="文件大小")
    chunk_count = Column(Integer, default=0, comment="分块数")
    status = Column(String(20), default="pending", comment="状态")
    error_message = Column(Text, nullable=True, comment="错误信息")

    # ===== 知识库分组 =====
    kb_id = Column(String(64), default="default", nullable=False, index=True, comment="知识库ID")

    # ===== 解析与切分留痕 =====
    content_hash = Column(String(64), nullable=True, index=True, comment="文件内容SHA256(去重用)")
    doc_type = Column(String(32), nullable=True, comment="文档类型")
    doc_type_source = Column(String(32), nullable=True, comment="类型判定来源")
    chunk_strategy = Column(String(64), nullable=True, comment="实际使用的切分策略")
    strategy_confidence = Column(String(16), nullable=True, comment="策略置信度")
    structure_detected = Column(String(128), nullable=True, comment="探测到的结构")

    created_at = Column(DateTime, default=func.now(), comment="上传时间")
    processed_at = Column(DateTime, nullable=True, comment="处理完成时间")


# ===== 法规版本与条文（P1）=====
#
# 这两张表把"版本与时效"从**文档里的概念**变成**数据库里的数据**。
#
# 为什么要它们（见 docs/11-文档处理管线.md §五、docs/09-当前状态对账.md §六）：
#   建表之前，条文的原文与元数据**只存在向量库里** —— 向量库本该是"可重建的
#   派生数据"，却兼任了"原文唯一副本"。结果两件事都做不好：
#     - Milvus 数据丢了，元数据（版本/时效）无从恢复
#     - 引用白名单校验只能查 Milvus，而那里可能已被污染
#   建完之后：MySQL 是权威源，Milvus 退回索引角色。
#
# 字段严格按 docs/02-设计文档.md §3.1，不加不减；要加字段属设计变更，须先改文档。


class LawVersion(Base):
    """法规版本表 —— 同一部法的多个版本靠 `law_name` 归组。

    本项目的两条差异化能力都依赖它：
      - **时效过滤**（需求 F4.2）：`validity_status` 决定检索时是否召回该版本
      - **引用标注**（需求 F3.3）：`version_label` 让引用处能显示"2012年修正 · 现行有效"

    `document_id` 指向 `documents.id`，即"这个版本来自哪份上传的文件"。
    """
    __tablename__ = "law_versions"

    id = Column(String(64), primary_key=True, comment="版本ID")
    document_id = Column(String(64), nullable=False, index=True, comment="来源文档ID（→documents.id）")
    law_name = Column(String(200), nullable=False, index=True, comment="法名，同一部法的多个版本靠它归组")
    law_level = Column(
        Enum(
            "宪法", "法律", "行政法规", "地方性法规", "司法解释", "部门规章", "其他",
            name="law_level_enum",
        ),
        nullable=True,
        comment="法律层级",
    )
    issuing_body = Column(String(200), nullable=True, comment="制定机关")
    doc_no = Column(String(100), nullable=True, comment="发文字号")
    version_label = Column(String(100), nullable=True, comment='版本标签，如 "2007年通过" / "2012年修正"')
    publish_date = Column(Date, nullable=True, comment="公布日期")
    effective_date = Column(Date, nullable=True, comment="施行日期（判断历史法条适用哪一版看它）")
    expiry_date = Column(Date, nullable=True, comment="失效日期（NULL = 现行有效）")
    validity_status = Column(
        Enum("effective", "amended", "repealed", "pending", name="validity_status_enum"),
        nullable=False,
        default="effective",
        index=True,
        comment="时效状态：effective/amended/repealed/pending",
    )
    superseded_by = Column(String(64), nullable=True, comment="被哪个版本取代（→law_versions.id）")
    source_url = Column(String(500), nullable=True, comment="来源链接")


class LawArticle(Base):
    """条文表 —— **权威源（ground truth）**。

    向量库里的 chunk 是它的"检索索引"，不是权威副本：
      - 引用白名单校验查**本表**（需求 F2.8）
      - 换 embedding 模型/重建索引时，从本表重新生成，元数据不丢

    注意语义：这是**条文**表，来源是解析结果（`ParsedSection`），不是"切分出的 chunk 列表"。
    现阶段一条法条 = 一个 chunk，所以数量一致；将来非按条切分的文档（FAQ、降级文本）
    不应写进本表。
    """
    __tablename__ = "law_articles"

    id = Column(String(64), primary_key=True, comment="条文ID")
    version_id = Column(String(64), nullable=False, index=True, comment="所属版本（→law_versions.id）")
    article_no = Column(String(50), nullable=False, comment='条文号，如 "第四十七条"')
    article_no_norm = Column(Integer, nullable=True, comment="归一化条文号，如 47（用于排序/匹配）")
    chapter_path = Column(String(500), nullable=True, comment='章节路径，如 "第四章 劳动合同的解除和终止"')
    seq = Column(Integer, nullable=True, comment="在版本内的顺序号")
    content = Column(Text, nullable=False, comment="条文原文")

    __table_args__ = (
        Index("idx_version_article", "version_id", "article_no_norm"),
    )


# ===== 数据库管理器 =====
class DatabaseManager:
    """MySQL 数据库管理器"""

    def __init__(self):
        self._engine = None
        self._session_factory = None

    def init_db(self):
        """初始化数据库连接和表"""
        if self._engine is None:
            logger.info(
                "[Database] 连接 MySQL: %s:%s/%s",
                settings.MYSQL_HOST, settings.MYSQL_PORT, settings.MYSQL_DATABASE,
            )
            self._engine = create_engine(
                settings.mysql_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=settings.DEBUG,
            )
            self._session_factory = sessionmaker(bind=self._engine)
            Base.metadata.create_all(self._engine)
            logger.info("[Database] 数据库表创建完成")

    def get_session(self) -> Session:
        """获取数据库会话"""
        if self._session_factory is None:
            self.init_db()
        return self._session_factory()

    # ===== 会话操作 =====
    def create_conversation(self, title: str, model_name: str) -> Conversation:
        """创建新会话"""
        session = self.get_session()
        try:
            conv = Conversation(
                id=str(uuid.uuid4()),
                title=title,
                model_name=model_name,
            )
            session.add(conv)
            session.commit()
            session.refresh(conv)
            return conv
        finally:
            session.close()

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取会话"""
        session = self.get_session()
        try:
            return session.query(Conversation).filter_by(id=conversation_id).first()
        finally:
            session.close()

    def list_conversations(self, limit: int = 20, offset: int = 0) -> List[Conversation]:
        """获取会话列表"""
        session = self.get_session()
        try:
            return (
                session.query(Conversation)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
        finally:
            session.close()

    def delete_conversation(self, conversation_id: str):
        """删除会话（级联删除消息）"""
        session = self.get_session()
        try:
            conv = session.query(Conversation).filter_by(id=conversation_id).first()
            if conv:
                session.delete(conv)
                session.commit()
        finally:
            session.close()

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        """重命名会话(F5.1)。返回是否命中并更新。"""
        session = self.get_session()
        try:
            conv = session.query(Conversation).filter_by(id=conversation_id).first()
            if not conv:
                return False
            conv.title = title
            session.commit()
            return True
        finally:
            session.close()

    # ===== 消息操作 =====
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model_name: str = None,
        tokens_used: int = 0,
        retrieval_info: dict = None,
        duration_ms: int = None,
        first_token_ms: int = None,
    ) -> Message:
        """添加消息"""
        session = self.get_session()
        try:
            msg = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                model_name=model_name,
                tokens_used=tokens_used,
                retrieval_info=retrieval_info,
                duration_ms=duration_ms,
                first_token_ms=first_token_ms,
            )
            session.add(msg)
            session.commit()
            msg_id = msg.id

            # 更新会话的更新时间
            conv = session.query(Conversation).filter_by(id=conversation_id).first()
            if conv:
                conv.updated_at = datetime.now()
                session.commit()

            # 重新查询返回完整的消息对象
            return session.query(Message).filter_by(id=msg_id).first()
        finally:
            session.close()

    def get_messages(self, conversation_id: str, limit: int = 50) -> List[Message]:
        """获取会话消息列表"""
        session = self.get_session()
        try:
            return (
                session.query(Message)
                .filter_by(conversation_id=conversation_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def get_message_count(self, conversation_id: str) -> int:
        """获取会话消息数量"""
        session = self.get_session()
        try:
            return session.query(Message).filter_by(conversation_id=conversation_id).count()
        finally:
            session.close()

    # ===== 文档操作 =====
    def add_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        *,
        original_filename: str = None,
        kb_id: str = "default",
        content_hash: str = None,
        doc_type: str = None,
        doc_type_source: str = None,
        chunk_strategy: str = None,
        strategy_confidence: str = None,
        structure_detected: str = None,
    ) -> Document:
        """添加文档记录。"""
        session = self.get_session()
        try:
            doc = Document(
                id=doc_id,
                filename=filename,
                original_filename=original_filename or filename,
                file_type=file_type,
                file_size=file_size,
                kb_id=kb_id,
                content_hash=content_hash,
                doc_type=doc_type,
                doc_type_source=doc_type_source,
                chunk_strategy=chunk_strategy,
                strategy_confidence=strategy_confidence,
                structure_detected=structure_detected,
            )
            session.add(doc)
            session.commit()
            return doc
        finally:
            session.close()

    def get_document(self, doc_id: str) -> Optional[Document]:
        """按 ID 取文档。"""
        session = self.get_session()
        try:
            return session.query(Document).filter_by(id=doc_id).first()
        finally:
            session.close()

    def find_document_by_hash(self, kb_id: str, content_hash: str) -> Optional[Document]:
        """按内容哈希在同一知识库内查重(F1.8)。"""
        if not content_hash:
            return None
        session = self.get_session()
        try:
            return (
                session.query(Document)
                .filter_by(kb_id=kb_id, content_hash=content_hash)
                .filter(Document.status == "completed")
                .first()
            )
        finally:
            session.close()

    def delete_document(self, doc_id: str) -> bool:
        """删除文档记录。"""
        session = self.get_session()
        try:
            doc = session.query(Document).filter_by(id=doc_id).first()
            if not doc:
                return False
            session.delete(doc)
            session.commit()
            return True
        finally:
            session.close()

    def update_document_status(self, doc_id: str, status: str, chunk_count: int = 0, error_message: str = None):
        """更新文档状态"""
        session = self.get_session()
        try:
            doc = session.query(Document).filter_by(id=doc_id).first()
            if doc:
                doc.status = status
                doc.chunk_count = chunk_count
                doc.error_message = error_message
                if status == "completed":
                    doc.processed_at = datetime.now()
                session.commit()
        finally:
            session.close()

    # ===== 条文操作（P1 / T-25）=====
    def upsert_law_articles(self, rows: list[dict]) -> int:
        """写入/更新一份文档的全部条文，**一次事务**。返回写入条数。

        为什么和 `documents` 分开调用却仍要说是"一次事务"：
            本方法内部自成事务（commit 一次）—— 要么这份文件的条文全进去，
            要么一条都不进去。半截写入会让 `law_articles` 与 `documents.chunk_count`
            对不上，而这种不一致事后很难查。

        为什么用 upsert 而不是纯 insert：
            条文 ID 是确定性的（`make_article_id`），所以**重新入库**时
            同一份文件的同一条法条主键不变 —— upsert 保证不会产生重复条文。

        rows 每项：id / version_id / article_no / article_no_norm / chapter_path / seq / content
        """
        if not rows:
            return 0
        session = self.get_session()
        try:
            for row in rows:
                existing = session.get(LawArticle, row["id"])
                if existing is None:
                    session.add(LawArticle(**row))
                else:
                    for key, value in row.items():
                        setattr(existing, key, value)
            session.commit()
            return len(rows)
        finally:
            session.close()

    def count_law_articles(self, version_id: str = None) -> int:
        """统计条文数（可按版本过滤）。上线前自检用。"""
        session = self.get_session()
        try:
            query = session.query(LawArticle)
            if version_id:
                query = query.filter(LawArticle.version_id == version_id)
            return query.count()
        finally:
            session.close()

    def existing_law_article_ids(self, ids) -> set:
        """这批条文 ID 里，哪些在 `law_articles` 里真实存在（引用白名单校验用，T-29）。

        一次 `IN` 查询而不是逐条 `get`：一次回答里的引用通常 1~5 条，
        但校验函数签名叫批量，就该真的批量 —— 否则以后接上更多引用时会变成 N 次往返。

        为什么查的是主键而不是 (version_id, article_no) 组合：
            主键由 `make_article_id` 确定性生成，查主键既最快，又只有一个事实来源
            （换成组合条件就等于把 ID 规则抄了第二遍，早晚会漂移）。
        """
        wanted = [i for i in (ids or []) if i]
        if not wanted:
            return set()
        session = self.get_session()
        try:
            rows = session.query(LawArticle.id).filter(LawArticle.id.in_(wanted)).all()
            return {row[0] for row in rows}
        finally:
            session.close()

    def law_version_exists(self, version_id: str) -> bool:
        """版本是否已灌种子。

        入库时用它留痕：若 `law_articles.version_id` 指向一个 `law_versions` 里
        不存在的版本，时效过滤会**静默失效**（join 不上 → 过滤条件永远不满足）。
        这类"不报错但功能没了"的情况必须当场出声。
        """
        if not version_id:
            return False
        session = self.get_session()
        try:
            return session.query(LawVersion).filter_by(id=version_id).count() > 0
        finally:
            session.close()

    def list_documents(
        self,
        limit: int = 20,
        offset: int = 0,
        kb_id: str = None,
    ) -> List[Document]:
        """获取文档列表(可按知识库过滤)。"""
        session = self.get_session()
        try:
            query = session.query(Document)
            if kb_id:
                query = query.filter(Document.kb_id == kb_id)
            return (
                query.order_by(Document.created_at.desc()).limit(limit).offset(offset).all()
            )
        finally:
            session.close()

    def count_documents(self, kb_id: str = None) -> int:
        """统计文档数。"""
        session = self.get_session()
        try:
            query = session.query(Document)
            if kb_id:
                query = query.filter(Document.kb_id == kb_id)
            return query.count()
        finally:
            session.close()

    def ping(self) -> bool:
        """检查数据库连接"""
        try:
            session = self.get_session()
            session.execute(func.now())
            session.close()
            return True
        except Exception:
            return False


# 全局实例
db_manager = DatabaseManager()
