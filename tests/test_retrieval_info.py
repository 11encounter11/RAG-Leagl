# -*- coding: utf-8 -*-
"""检索留痕测试（T-31，需求 F8.2）。

守的是两件事：

1. **落库的留痕里不许有条文全文** —— 原文的正本是 `law_articles`，凭
   `version_id + 条文号 + section_seq` 随时捞得回来。在 messages 里再抄一份，
   等于造第二个副本兼任权威源（T-23 建权威表就是为了消灭这种局面）。
   `build_sources` 返回的字典里**就带着 `"text"` 全文**，所以这条必须钉住：
   哪天有人图省事把整份 sources 塞进去，这里要立刻红。

2. **`_persist` 真的把留痕交给了写库函数** —— 这个字段此前恒为 NULL，
   原因正是"列和 add_message 的参数都在，只是没人传"。所以光测构造函数不够，
   必须测到调用点。

全部脱网可跑：不连 MySQL，写库用桩替换。
"""
from __future__ import annotations

import json

import pytest

from app.config.settings import settings
from app.core.law_versions import make_article_id
from app.core.rag.embedder import EMBEDDING_MODEL
from app.services import chat_service as cs
from app.services.chat_service import ChatService, build_retrieval_info


def _source(index=1, **overrides):
    """一条真实形态的来源（字段取自 build_sources，含全文，用来验证它不会被带进留痕）。"""
    base = {
        "index": index,
        "node_id": "node-abc",
        "document_id": "doc-001",
        "doc_type": "statute",
        "law_name": "中华人民共和国劳动法",
        "filename": "中华人民共和国劳动法_20181229.docx",
        "article_no": "第八十二条",
        "article_no_norm": 82,
        "chapter_path": "第十章 劳动争议",
        "clause_no": "",
        "version_id": "90ed9065-6212-5eb5-b39d-4938e2a77d76",
        "section_seq": 81,
        "validity_status": "effective",
        "validity_label": "现行有效",
        "version_label": "2018年修正",
        "region_code": "",
        "sheet": "",
        "row": "",
        "score_raw": 0.0,
        "score_normalized": 0.6231,
        "score": 0.6231,
        "search_preview": "第八十二条　提出仲裁要求的一方应当自劳动争议发生之日起...",
        "text_preview": "第八十二条　提出仲裁要求的一方应当自劳动争议发生之日起...",
        "text": (
            "第八十二条　提出仲裁要求的一方应当自劳动争议发生之日起六十日内向"
            "劳动争议仲裁委员会提出书面申请；当事人因不可抗力或者有其他正当理由"
            "超过规定的申请期限的，仲裁委员会应当受理。"
        ),
    }
    base.update(overrides)
    return base


def _retrieval(**overrides):
    base = {
        "nodes": [],
        "context": "",
        "sources": [_source()],
        "workflow_steps": ["retrieval:direct"],
        "reflection_score": None,
        "reflection_note": "",
        "refused": False,
        "refuse_reason": "",
        "domain_score": 0.6231,
        "confidence_level": "high",
    }
    base.update(overrides)
    return base


# ============================================================
# 1) 不许带全文
# ============================================================
def test_retrieval_info_drops_article_full_text():
    info = build_retrieval_info(_retrieval(), model_name="deepseek-flash", top_k=5)
    dumped = json.dumps(info, ensure_ascii=False)

    for forbidden in ("提出仲裁要求的一方", "劳动争议发生之日起六十日内", "node-abc"):
        assert forbidden not in dumped, f"留痕里出现了不该有的内容: {forbidden}"


def test_retrieval_info_keeps_fields_needed_to_recover_the_original():
    """全文不存，但要保证存下来的键能把原文捞回来。"""
    traced = build_retrieval_info(_retrieval(), model_name="deepseek-flash", top_k=5)[
        "sources"
    ][0]

    assert traced["law_name"] == "中华人民共和国劳动法"
    assert traced["article_no"] == "第八十二条"
    assert traced["version_id"] == "90ed9065-6212-5eb5-b39d-4938e2a77d76"
    assert traced["section_seq"] == 81
    assert traced["validity_status"] == "effective"
    assert traced["score"] == 0.6231


def test_traced_article_id_matches_the_authoritative_primary_key():
    """留痕里的 article_id 必须与入库时写进 law_articles 的主键是同一套算法。

    另立一套 ID 规则的话，两边迟早漂移，而漂移的表现是"拿着留痕查不到那一行"——
    复盘时才发现，等于没有留痕。
    """
    traced = build_retrieval_info(_retrieval(), model_name="deepseek-flash", top_k=5)[
        "sources"
    ][0]

    assert traced["article_id"] == make_article_id(
        "90ed9065-6212-5eb5-b39d-4938e2a77d76", "第八十二条", 81
    )


# ============================================================
# 2) 配置快照
# ============================================================
def test_retrieval_info_snapshots_the_switches_in_effect(monkeypatch):
    """同一份召回结果，开着/关着时效过滤是两回事，不留配置就读不出含义。"""
    monkeypatch.setattr(settings, "ENABLE_VALIDITY_FILTER", False, raising=True)
    monkeypatch.setattr(settings, "ENABLE_RERANK", True, raising=True)

    config = build_retrieval_info(
        _retrieval(), model_name="deepseek-flash", top_k=3
    )["config"]

    assert config["validity_filter"] is False
    assert config["rerank"] is True
    assert config["top_k"] == 3
    assert config["model"] == "deepseek-flash"
    # 没传阈值时记的是**实际生效值**(退回服务端标定值), 不是 null —— 留痕不能留个歧义让人猜
    assert config["similarity_threshold"] == settings.SIMILARITY_THRESHOLD
    # 阈值也要进快照：拒答闸门余量只有 0.0105，事后复盘必须知道当时定在哪
    assert config["domain_score_reject"] == pytest.approx(settings.DOMAIN_SCORE_REJECT)


def test_snapshot_names_the_threshold_yardstick():
    """P2-20: 留痕必须说清这个数是拿什么尺子量的, 否则两个尺度同数不同义。

    `similarity_threshold` 比的是重排归一化相关度(0~1), 用法是只判本批 top1。
    """
    config = build_retrieval_info(_retrieval(), model_name="m", top_k=5)["config"]

    assert config["similarity_threshold_scale"] == "rerank_relevance"
    assert config["similarity_threshold_rule"] == "top1_gate"


def test_retrieval_info_records_the_actual_threshold_gate():
    """配置里有值不等于这次真的比过 —— 实际闸门状态单独记。"""
    usage = {
        "scale": "rerank_relevance",
        "rule": "top1_gate",
        "value": 0.1,
        "applied": True,
        "top1_score": 0.1954,
        "candidates": 5,
        "kept": 5,
        "returned": 5,
    }
    info = build_retrieval_info(
        _retrieval(threshold_usage=usage), model_name="m", top_k=5
    )

    assert info["threshold_usage"] == usage


def test_missing_threshold_usage_is_an_empty_object_not_null():
    """旧调用方/直连路径可能拿不到实况; 写空对象, 别写 null 让查询分两派。"""
    info = build_retrieval_info(_retrieval(), model_name="m", top_k=5)
    assert info["threshold_usage"] == {}


def test_snapshot_keeps_the_request_level_cutoff_threshold():
    """`similarity_threshold` 由请求体带进来，且真的会改变闸门结果
    （`filter_by_similarity_threshold`：top1 低于它就整批不用）。
    同一个问题换个阈值，召回条数就不同 —— 不记它，"这次怎么一条都没回来"就没法解释。
    """
    info = build_retrieval_info(
        _retrieval(), model_name="deepseek-flash", top_k=5, similarity_threshold=0.5
    )
    assert info["config"]["similarity_threshold"] == 0.5


def test_snapshot_records_the_embedding_model_actually_in_use():
    """`.env` 里的 `EMBEDDING_MODEL_NAME=BAAI/bge-large-zh-v1.5` 是**死配置** ——

    `embedder.py` 根本没读它，实际请求的一直是智谱 embedding-3。留痕去读 settings
    就会把"没用过的模型"记成"用过的"，而这恰好是最要命的一项：嵌入模型一变，
    全部向量作废，复盘时靠的就是这一行。所以这里钉住"记的是真在用的那个"。
    """
    config = build_retrieval_info(
        _retrieval(), model_name="deepseek-flash", top_k=5
    )["config"]

    assert config["embedding_model"] == EMBEDDING_MODEL == "embedding-3"


def test_retrieval_info_records_the_refusal_path():
    """拒答那次没调 LLM，留痕是唯一能说明"为什么拒"的记录。"""
    info = build_retrieval_info(
        _retrieval(
            sources=[],
            refused=True,
            refuse_reason="向量最高相似度 0.31 低于拒答阈值",
            confidence_level="out_of_domain",
        ),
        model_name="deepseek-flash",
        top_k=5,
    )

    assert info["refused"] is True
    assert info["refuse_reason"] == "向量最高相似度 0.31 低于拒答阈值"
    assert info["retrieval_count"] == 0
    assert info["sources"] == []


def test_build_retrieval_info_tolerates_missing_keys():
    """工作流/直接检索两条路的字段齐全度不同，留痕不能因此抛异常 ——
    留痕失败不该把整次问答带崩。"""
    info = build_retrieval_info({"sources": [{"index": 1}]}, model_name="m", top_k=5)

    assert info["schema"] == "retrieval_info.v1"
    assert info["sources"][0]["version_id"] == ""
    assert info["sources"][0]["article_id"] is None  # 无 version_id = 无法核验, 不是报错
    assert info["confidence_level"] == "high"


@pytest.mark.parametrize("empty", [None, {}])
def test_build_retrieval_info_handles_empty_retrieval(empty):
    info = build_retrieval_info(empty, model_name="m", top_k=5)
    assert info["sources"] == []
    assert info["retrieval_count"] == 0


# ============================================================
# 3) 调用点：留痕真的交给了写库函数
# ============================================================
class _FakeMessage:
    id = 77


class _FakeDb:
    """记录每次 add_message 的入参，用来断言留痕真的被传下去了。"""

    def __init__(self):
        self.calls = []

    def add_message(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeMessage()


def test_persist_passes_retrieval_info_to_add_message(monkeypatch):
    """这条是 T-31 的本体。

    此前的失效方式正是"db_store 收这个参数、chat_service 不传"，
    所以必须测到调用点，而不是只测构造函数。
    """
    fake = _FakeDb()
    monkeypatch.setattr(cs, "db_manager", fake, raising=True)

    ChatService()._persist("conv-1", "仲裁时效多久", "一年。[1]", "deepseek-flash", _retrieval(), 5)

    assert [c["role"] for c in fake.calls] == ["user", "assistant"]
    user_row, assistant_row = fake.calls

    # 用户消息没有留痕可言，不该被塞一份假的
    assert "retrieval_info" not in user_row

    info = assistant_row["retrieval_info"]
    assert isinstance(info, dict), "retrieval_info 必须是 dict(JSON 列), 不是字符串"
    assert info["sources"][0]["article_no"] == "第八十二条"
    assert "提出仲裁要求的一方" not in json.dumps(info, ensure_ascii=False)


def test_persist_still_writes_info_when_retrieval_absent(monkeypatch):
    """拒答分支之外的任何调用都不能把字段留成 NULL —— 那等于回到 T-31 之前。"""
    fake = _FakeDb()
    monkeypatch.setattr(cs, "db_manager", fake, raising=True)

    ChatService()._persist("conv-1", "问题", "回答", "deepseek-flash")

    info = fake.calls[1]["retrieval_info"]
    assert info["sources"] == []
    assert info["retrieval_count"] == 0
    assert "config" in info
