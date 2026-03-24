import logging
from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field

from demo.utils.http import get_debug_http_client

logger = logging.getLogger(__name__)


class PrivateAPIReranker(BaseNodePostprocessor):
    """纯异步的企业私有 Reranker 客户端"""

    api_url: str = Field(description="Rerank API 地址")
    app_token: str = Field(description="认证 Token")
    app_id: str = Field(description="应用 ID")
    biz_no: str = Field(description="业务编码")
    model: str = Field(description="模型名称")
    top_n: int = Field(default=3, description="最终截取数量")
    timeout: int = Field(default=30, description="网络请求超时时间")

    @classmethod
    def class_name(cls) -> str:
        return "PrivateAPIReranker"

    @classmethod
    def from_defaults(
        cls, private_ext, top_n: Optional[int] = None
    ) -> Optional["PrivateAPIReranker"]:
        from demo.config import get_settings  # 延迟导入，防止循环依赖

        config = get_settings()

        # 兼容之前讨论的分层配置
        # private_ext = getattr(config.llm, "provider_private", None)
        rerank_url = (
            private_ext.rerank_api_url
            if private_ext
            else getattr(config.llm, "rerank_api_url", None)
        )

        if not rerank_url:
            logger.warning("未配置 Rerank API URL，将跳过重排步骤。")
            return None

        return cls(
            api_url=rerank_url,
            app_token=private_ext.app_token if private_ext else config.llm.api_key,
            app_id=private_ext.app_id if private_ext else "",
            biz_no=private_ext.biz_no if private_ext else "",
            model=config.llm.rerank_model,
            top_n=top_n or config.rag.similarity_top_k,
            timeout=config.llm.timeout,
        )

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        logger.warning(
            "⚠️ 警告：正在同步调用 Reranker，高并发下会阻塞！请确保使用 ainvoke/aquery。"
        )
        return nodes[: self.top_n]

    async def _apostprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """使用 AsyncClient 实现非阻塞网络调用"""
        if not query_bundle or not nodes:
            return nodes

        payload = {
            "appId": self.app_id,
            "bizNo": self.biz_no,
            "model": self.model,
            "query": query_bundle.query_str,
            "documents": [node.node.get_content() for node in nodes],
        }
        headers = {"app-token": self.app_token, "content-type": "application/json"}

        try:
            async with get_debug_http_client(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url, json=payload, headers=headers
                )
                response.raise_for_status()
                data = response.json()

            if str(data.get("code")) != "0000" or data.get("result") != 1:
                logger.error(f"Rerank API 业务报错: {data.get('message')}")
                return nodes[: self.top_n]

            # 映射得分并重排
            reranked_nodes = []
            for res in data.get("content", {}).get("results", []):
                idx = res["index"]
                node_with_score = nodes[idx]
                node_with_score.score = res["relevance_score"]
                reranked_nodes.append(node_with_score)

            reranked_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            return reranked_nodes[: self.top_n]

        except Exception as e:
            logger.error(f"重排 API 异常: {e}。触发降级，返回原节点。")
            return nodes[: self.top_n]
