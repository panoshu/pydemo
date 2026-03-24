import pytest
from pydantic import ValidationError

from demo.config.loader import AppSettings


# ==========================================
# Fixture: 强制隔离测试环境 (极其重要)
# ==========================================
@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """
    自动隔离测试环境！
    将工作目录切换到临时空文件夹，防止读取到开发者本地真实的 config.dev.yml 或 .env。
    确保测试结果的绝对纯净和稳定。
    """
    monkeypatch.chdir(tmp_path)


# ==========================================
# 辅助函数：注入使系统能勉强启动的“最基础必填环境变量”
# ==========================================
def inject_base_required_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB__HOST", "localhost")
    monkeypatch.setenv("DB__PORT", "3306")
    monkeypatch.setenv("DB__USER", "root")
    monkeypatch.setenv("DB__PASSWORD", "secret")
    monkeypatch.setenv("API__URL", "http://test")
    monkeypatch.setenv("LLM__BASE_URL", "http://local:8000/v1")
    monkeypatch.setenv("LLM__API_KEY", "sk-dummy")
    monkeypatch.setenv("LOG__LEVEL", "INFO")

    # 【修复】：必须注入新模块 rag 的触发器，否则系统会报 rag 模块缺失
    monkeypatch.setenv("RAG__CHUNK_SIZE", "512")


# ==========================================
# 测试用例
# ==========================================
def test_missing_required_config_fails():
    """测试：如果没有提供完整的必填配置，程序必须立刻崩溃 (Fail-Fast)"""
    with pytest.raises(ValidationError) as exc_info:
        AppSettings.load()

    # 此时在 isolated_env 沙盒下，所有配置都会报缺失
    error_msg = str(exc_info.value)
    assert "db\n  Field required" in error_msg
    assert "api\n  Field required" in error_msg
    assert "llm\n  Field required" in error_msg
    assert "rag\n  Field required" in error_msg


def test_invalid_config_values_fails(monkeypatch: pytest.MonkeyPatch):
    """测试：充血模型校验生效，提供非法值（如端口越界）必须被拦截"""
    inject_base_required_env(monkeypatch)

    # 故意设置一个错误的端口 (有效范围 1-65535)
    monkeypatch.setenv("DB__PORT", "99999")

    with pytest.raises(ValidationError) as exc_info:
        AppSettings.load()

    # 【修复】：断言 Pydantic V2 针对 le (小于等于) 约束的真实原生报错信息
    assert "Input should be less than or equal to 65535" in str(exc_info.value)


def test_env_variable_override_and_defaults(monkeypatch: pytest.MonkeyPatch):
    """测试：环境变量能正确覆盖嵌套配置，且默认值正常生成"""
    inject_base_required_env(monkeypatch)

    # 覆盖嵌套字典的值
    monkeypatch.setenv("DB__HOST", "192.168.1.100")
    monkeypatch.setenv("FEATURE__ENABLE_NEW_UI", "true")
    monkeypatch.setenv("FEATURE__MAX_EXPORT_LIMIT", "5000")

    # 覆盖新增的 LLM 字段
    monkeypatch.setenv("LLM__LLM_MODEL", "qwen-max-test")

    config = AppSettings.load()

    # 1. 验证覆盖是否成功
    assert config.db.host == "192.168.1.100"
    assert config.feature.enable_new_ui is True
    assert config.feature.max_export_limit == 5000
    assert config.llm.llm_model == "qwen-max-test"

    # 2. 验证由于没有被覆盖，默认值是否保持原样
    assert config.rag.chunk_size == 512
    assert config.rag.similarity_top_k == 3
