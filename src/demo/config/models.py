from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

# ==========================================
# 1. 提取并定义可复用的带校验的类型
# ==========================================
# 定义一个 Port 类型：必须是整数，且 >=1 且 <=65535
PortType = Annotated[
    int, Field(ge=1, le=65535, description="网络端口号必须在 1-65535 之间")
]

# 非空字符串
NonEmptyStr = Annotated[str, Field(min_length=1)]


# ==========================================
# 2. 配置模型定义
# ==========================================
class DatabaseConfig(BaseModel):
    host: NonEmptyStr
    port: PortType  # <--- 没有任何冗长的 @field_validator，只需声明类型
    user: str
    password: str


class ApiConfig(BaseModel):
    url: NonEmptyStr
    timeout: int = 30


class PrivateApiCredentials(BaseModel):
    """私有化平台专属鉴权字段"""

    app_token: str = ""
    app_id: str = ""
    biz_no: str = ""

    embed_api_url: str
    rerank_api_url: str


class LlmConfig(BaseModel):
    # 通用连接配置
    base_url: NonEmptyStr
    api_key: NonEmptyStr
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    context_window: int = 2 * 1024

    # 模型路由配置
    llm_model: NonEmptyStr = "qwen-2.5-72b"
    vlm_model: NonEmptyStr = "qwen-vl-max"
    embed_model: NonEmptyStr = "qwen3_embedding_4b"
    rerank_model: NonEmptyStr = "qwen3_reranker_4b"

    timeout: int = 2 * 60

    provider_private: Optional[PrivateApiCredentials] = None


class RagConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 3
    recall_multiplier: int = 2
    enable_bm25: bool = True
    chunk_strategy: Literal["sentence", "fixed_length", "markdown"] = "sentence"


class MilvusConfig(BaseModel):
    uri: str = "./database/milvus_lite.db"
    collection_name: str = "bid_compliance_rules"
    dim: int = 1024
    metric_type: Literal["L2", "IP", "COSINE"] = "COSINE"


class LogConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    file: str = "app.log"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5
    format: str = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"


class FeatureConfig(BaseModel):
    enable_new_ui: bool = False
    max_export_limit: int = 1000
    promotion_banner_text: str = "Welcome!"


class RedisConfig(BaseModel):
    host: NonEmptyStr
    port: PortType = 6379
