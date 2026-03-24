# src/demo/llm/embedding.py
import logging
from typing import Any, List

import httpx
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class PrivateAPIEmbedding(BaseEmbedding):
    """高度定制化的企业私有 Embedding 客户端"""

    api_url: str
    app_token: str
    app_id: str
    biz_no: str
    model_name: str

    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "PrivateAPIEmbedding"

    @classmethod
    def from_defaults(
        cls, private_ext, embed_batch_size: int = 10
    ) -> "PrivateAPIEmbedding":
        """【DDD 充血模型】封装 Embedding 客户端的初始化"""
        from demo.config import get_settings

        config = get_settings()

        # private_ext = getattr(config.llm, "provider_private", None)

        return cls(
            api_url=private_ext.embed_api_url
            if private_ext
            else getattr(config.llm, "embed_api_url", ""),
            app_token=private_ext.app_token if private_ext else config.llm.api_key,
            app_id=private_ext.app_id if private_ext else "",
            biz_no=private_ext.biz_no if private_ext else "",
            model_name=config.llm.embed_model,
            embed_batch_size=embed_batch_size,
            timeout=config.llm.timeout,
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aget_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        res = await self._aget_text_embeddings([text])
        return res[0]

    def _build_payload_and_headers(self, texts: List[str]):
        print(self.app_token, self.app_id)
        headers = {"app-token": self.app_token, "content-type": "application/json"}
        payload = {
            "appId": self.app_id,
            "bizNo": self.biz_no,
            "input": texts,
            "model": self.model_name,
            "encoding_format": "float",
        }
        return headers, payload

    def _parse_response(self, data: dict) -> List[List[float]]:
        # 严格校验业务状态码
        if str(data.get("code")) != "0000" or data.get("result") != 1:
            raise ValueError(f"私有 Embedding API 业务报错: {data.get('message')}")

        content_data = data.get("content", {}).get("data", [])

        # 确保按照输入顺序返回 embedding
        content_data.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in content_data]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers, payload = self._build_payload_and_headers(texts)
        print("headers: ", headers, "payload: ", payload)
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                return self._parse_response(response.json())
        except Exception as e:
            logger.error(f"❌ 私有化 Embedding API 调用失败: {e}")
            raise

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers, payload = self._build_payload_and_headers(texts)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.api_url, json=payload, headers=headers
                )
                response.raise_for_status()
                return self._parse_response(response.json())
        except Exception as e:
            logger.error(f"❌ 异步私有化 Embedding API 调用失败: {e}")
            raise
