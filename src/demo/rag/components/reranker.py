# src/demo/rag/components/reranker.py
import logging
import uuid
from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import BaseModel, Field, PrivateAttr

from demo.config import get_settings
from demo.config.factory import _get_private_body_params, _get_private_headers
from demo.utils.http_client import http_client

logger = logging.getLogger(__name__)


# ==========================================
# Pydantic 接口模型定义
# ==========================================
class RerankerRequest(BaseModel):
    appId: str = ""
    bizNo: str = ""
    model: str
    query: str
    documents: List[str]


class RerankResultItem(BaseModel):
    index: int
    relevance_score: float


class RerankContent(BaseModel):
    results: List[RerankResultItem]


class PrivateRerankerResponse(BaseModel):
    result: int
    code: str
    message: str
    content: Optional[RerankContent] = None


# ==========================================
# 企业级 Reranker 适配器
# ==========================================
class PrivateAPIReranker(BaseNodePostprocessor):
    top_n: int = Field(default=3, description="重排后保留的节点数")
    _api_url: str = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = get_settings()
        if config.llm.mode == "private":
            assert config.llm.private_api is not None
            self._api_url = config.llm.private_api.rerank_api_url

    # ==================== 核心：纯逻辑复用层 ====================
    def _build_payload(self, nodes: List[NodeWithScore], query_str: str) -> dict:
        config = get_settings()
        base_params = _get_private_body_params()
        req = RerankerRequest(
            appId=base_params.get("appId", "default_app_id"),
            bizNo=base_params.get("bizNo", uuid.uuid4().hex),
            model=config.llm.rerank_model,
            query=query_str,
            documents=[node.node.get_content() for node in nodes],
        )
        return req.model_dump(exclude_none=True)

    def _parse_response(
        self, response_json: dict, original_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        resp_obj = PrivateRerankerResponse.model_validate(response_json)

        if resp_obj.code != "0000" or not resp_obj.content:
            raise ValueError(f"业务异常 [{resp_obj.code}]: {resp_obj.message}")

        reranked_nodes = []
        for item in resp_obj.content.results:
            # 边界保护
            if 0 <= item.index < len(original_nodes):
                node_copy = original_nodes[item.index]
                node_copy.score = item.relevance_score
                reranked_nodes.append(node_copy)

        # 按照新分数降序排列，并截取 top_n
        reranked_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return reranked_nodes[: self.top_n]

    # ==================== I/O 层：双轨制执行 ====================
    async def _apostprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """异步轨道"""
        if not query_bundle or not nodes:
            return nodes
        try:
            # 1. 组装
            payload = self._build_payload(nodes, query_bundle.query_str)
            # 2. 异步 I/O (复用全局连接池)
            response = await http_client.async_client.post(
                self._api_url, headers=_get_private_headers(), json=payload
            )
            response.raise_for_status()
            # 3. 解析
            return self._parse_response(response.json(), nodes)
        except Exception as e:
            logger.error(f"❌ Async Rerank Error: {e}")
            return nodes[: self.top_n]

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """同步轨道"""
        if not query_bundle or not nodes:
            return nodes
        try:
            # 1. 组装
            payload = self._build_payload(nodes, query_bundle.query_str)
            # 2. 同步 I/O (随用随抛)
            with http_client.sync_session() as client:
                response = client.post(
                    self._api_url, headers=_get_private_headers(), json=payload
                )
                response.raise_for_status()
            # 3. 解析
            return self._parse_response(response.json(), nodes)
        except Exception as e:
            logger.error(f"❌ Sync Rerank Error: {e}")
            return nodes[: self.top_n]
