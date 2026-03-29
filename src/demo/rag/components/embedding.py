# src/demo/rag/components/embedding.py
import logging
import uuid
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import BaseModel, PrivateAttr

from demo.config import get_settings
from demo.config.factory import _get_private_body_params, _get_private_headers
from demo.utils.http_client import http_client

logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    appId: str = ""
    bizNo: str = ""
    input: List[str]
    model: str
    encoding_format: str = "float"


class EmbeddingDataItem(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"


class EmbeddingContent(BaseModel):
    data: List[EmbeddingDataItem]
    model: Optional[str] = None
    id: Optional[str] = None


class PrivateEmbeddingResponse(BaseModel):
    result: int
    code: str
    message: str
    content: Optional[EmbeddingContent] = None
    additions: Optional[dict] = None


# ==========================================
# 企业级 Embedding 适配器实现
# ==========================================
class PrivateAPIEmbedding(BaseEmbedding):
    """
    企业私有化部署 Embedding 模型适配器。
    基于 Pydantic 强类型约束，防范“伪200”报错。
    """

    _api_url: str = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(
        self,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        config = get_settings()
        self._model_name = config.llm.embed_model

        if config.llm.mode == "private":
            assert config.llm.private_api is not None
            self._api_url = config.llm.private_api.embed_api_url

    @classmethod
    def class_name(cls) -> str:
        return "PrivateAPIEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        embeddings = await self._aget_text_embeddings([query])
        return embeddings[0] if embeddings else []

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embeddings([text])[0]

    def _build_payload(self, texts: List[str]) -> dict:
        """纯函数：统一构造请求参数"""
        base_params = _get_private_body_params()
        req = EmbeddingRequest(
            appId=base_params.get("appId", "default_app_id"),
            bizNo=base_params.get("bizNo", uuid.uuid4().hex),
            input=texts,
            model=self._model_name,
        )
        return req.model_dump(exclude_none=True)

    def _parse_response(self, response_json: dict) -> List[List[float]]:
        """纯函数：统一校验并解析返回报文"""
        resp_obj = PrivateEmbeddingResponse.model_validate(response_json)

        # 拦截“伪 200”报错
        if resp_obj.code != "0000" or not resp_obj.content:
            raise ValueError(f"业务异常 [{resp_obj.code}]: {resp_obj.message}")

        # 提取并按 index 排序防乱序
        sorted_data = sorted(resp_obj.content.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步轨道 (走全局长连接池，速度极快)"""
        try:
            # 1. 组装
            payload = self._build_payload(texts)
            # 2. 异步 I/O
            response = await http_client.async_client.post(
                self._api_url, headers=_get_private_headers(), json=payload
            )
            response.raise_for_status()
            # 3. 解析
            return self._parse_response(response.json())
        except Exception as e:
            logger.error(f"❌ Async Embedding Error: {e}")
            return [[0.0] * 512 for _ in texts]  # 容错降级

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """同步轨道 (供独立建库脚本调用)"""
        try:
            # 1. 组装
            payload = self._build_payload(texts)
            # 2. 同步 I/O
            with http_client.sync_session() as client:
                response = client.post(
                    self._api_url, headers=_get_private_headers(), json=payload
                )
                response.raise_for_status()
            # 3. 解析
            return self._parse_response(response.json())
        except Exception as e:
            logger.error(f"❌ Sync Embedding Error: {e}")
            return [[0.0] * 512 for _ in texts]
