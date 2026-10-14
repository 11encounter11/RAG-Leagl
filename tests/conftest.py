"""pytest 共享夹具。

对应 Spring Boot 里的"测试配置类"—— 公共部分只写一次, 各测试文件直接用。
现在只有一个 `client`, 因为项目此前**完全没有 API 层测试**。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """API 测试客户端。

    刻意**不用** `with TestClient(app) as c` 的形式 —— 进入 with 块会触发
    lifespan(startup), 那会去连 MySQL/Redis/Milvus 并加载约 1 GB 的重排器模型,
    测试就没法脱网跑了。

    不带 `with` 的写法同样能发请求, 只是不跑 startup/shutdown —— 正合我们所需:
    API 契约测试要验证的是**路由与校验**, 不该依赖基础设施。
    """
    return TestClient(app)
