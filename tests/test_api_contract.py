# -*- coding: utf-8 -*-
"""API 契约测试（对应 `05-测试与评测.md` 的 L4 层）。

项目此前**完全没有 API 层测试** —— `09-当前状态对账.md` §六 把它列为遗留项。
这几条是起点，而且全部**脱网可跑**（依赖 `conftest.client`，不触发 lifespan）。

选这几条的理由：它们各自锁住一个"已经决定、以后不该被误改"的契约。
"""
from __future__ import annotations


def test_models_only_deepseek(client):
    """LLM 只保留 DeepSeek（2026-10-14 决定，智谱额度专供 embedding）。

    锁住这个决定：谁若把 GLM 加回路由，这条测试立刻失败。
    """
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json()["models"] == ["deepseek-v4.1-flash"]


def test_chat_rejects_removed_model(client):
    """带已下线的模型名 → 在**请求校验阶段**被拒（422），而不是等调用时才报错。

    422 这个状态码本身也是契约的一部分：说明校验发生在 FastAPI/pydantic 层。
    """
    resp = client.post(
        "/api/chat", json={"message": "你好", "model_name": "glm-5.3-flash"}
    )
    assert resp.status_code == 422


def test_chat_requires_message(client):
    """缺少 message 必须被拒。"""
    assert client.post("/api/chat", json={}).status_code == 422


def test_upload_rejects_unsupported_extension(client):
    """不支持的扩展名必须在**落盘之前**就被拒（400）。

    这条同时验证"门槛检查在前"这个顺序 —— 若哪天把校验挪到落盘之后，
    状态码可能还是 400，但磁盘上会多出垃圾文件。见 11-文档处理管线.md【0】。
    """
    resp = client.post(
        "/api/knowledge/upload", files={"file": ("a.exe", b"x")}
    )
    assert resp.status_code == 400
    assert "不支持的文件类型" in resp.json()["detail"]
