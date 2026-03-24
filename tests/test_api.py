from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from demo.api.dependencies import get_settings
from demo.config.loader import AppSettings
from demo.config.models import (
    ApiConfig,
    DatabaseConfig,
    FeatureConfig,
    LlmConfig,
    LogConfig,
    RagConfig,
)

# 导入 FastAPI 实例和配置相关的契约
from demo.main import app

client = TestClient(app)


# ==========================================
# Fixture 1: 拦截并注入虚假的全局配置
# ==========================================
@pytest.fixture(autouse=True)
def override_settings():
    """自动覆盖 FastAPI 的依赖注入，使用安全的 Mock 配置"""
    mock_config = AppSettings.model_construct(
        db=DatabaseConfig(host="mock-db", port=3306, user="u", password="p"),
        api=ApiConfig(url="http://mock-api", timeout=30),
        llm=LlmConfig(
            base_url="http://mock-llm:8000/v1",
            llm_model="mock-llm",
            vlm_model="mock-vlm",
            embed_model="mock-embed",
            api_key="mock-key",
            temperature=0.7,
            max_tokens=2048,
        ),
        rag=RagConfig(chunk_size=512, chunk_overlap=50, similarity_top_k=3),
        log=LogConfig(level="INFO", file="app.log"),
        feature=FeatureConfig(
            enable_new_ui=True,
            max_export_limit=999,
            promotion_banner_text="Test Banner",
        ),
    )

    app.dependency_overrides[get_settings] = lambda: mock_config
    yield
    app.dependency_overrides.clear()


# ==========================================
# Fixture 2: Mock LangGraph 避免真实的 LLM 调用
# ==========================================
@pytest.fixture
def mock_langgraph_workflow():
    """
    拦截 API 路由中的 build_workflow 调用。
    模拟 AI Agent 并发执行后的最终 State 返回结果。
    """
    with patch("demo.api.routers.review.build_workflow") as mock_build:
        # 构造一个虚假的 LangGraph App
        mock_graph_app = mock_build.return_value

        # 定义当执行 graph_app.invoke() 时返回的假数据
        mock_graph_app.invoke.return_value = {
            "grammar_issues": ["【语法】发现一处错别字。"],
            "logic_issues": [],
            "compliance_issues": ["【合规】响应时间违规！"],
            "vision_issues": [],
            "extracted_data": {"response_time": 999},
        }
        yield mock_graph_app


# ==========================================
# 测试用例
# ==========================================
def test_system_info_api():
    """测试系统信息接口 (验证 Mock 配置是否注入成功)"""
    response = client.get("/system-info")
    assert response.status_code == 200
    assert response.json()["db_host"] == "mock-db"


def test_business_feature_api():
    """测试业务功能开关接口"""
    response = client.get("/business/feature-status")
    data = response.json()

    assert response.status_code == 200
    assert data["export_limit"] == 999
    assert "新版 UI 已启用: Test Banner" in data["message"]


def test_ai_review_api(mock_langgraph_workflow):
    """
    测试 AI 标书审核入口 (验证 API 层的参数接收和结构化输出)
    注意这里使用了 mock_langgraph_workflow fixture 阻断了真实网络请求
    """
    payload = {
        "document_text": "这是一份包含错别字和合规问题的标书。",
        "images_base64": [],
    }

    response = client.post("/api/review", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "success"
    # 验证 API 是否正确汇总了假数据中的问题
    assert data["has_issues"] is True
    assert len(data["issues"]) == 2
    assert "【合规】响应时间违规！" in data["issues"]
    assert data["extracted_data"]["response_time"] == 999
