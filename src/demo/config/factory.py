# src/demo/llm/factory.py
import logging
import uuid
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from llama_index.llms.openai_like import OpenAILike
from pydantic import SecretStr

from demo.config import get_settings
from demo.utils.http_client import http_client

logger = logging.getLogger(__name__)


_config = get_settings()
_llm_cfg = _config.llm
_private_ext = getattr(_llm_cfg, "private_api", None)


def _get_private_headers() -> Dict[str, str]:

    headers = {}
    if _private_ext:
        headers.update(
            {
                "app-token": _private_ext.app_token,
            }
        )
    return headers


def _get_private_body_params() -> Dict[str, Any]:

    params = {}
    if _private_ext:
        if _private_ext.biz_no:
            params["appId"] = _private_ext.app_id
            params["bizNo"] = str(uuid.uuid4())

    # 如果还需要强制透传 do_sample 等参数，也可以在这里加
    # params["do_sample"] = False
    return params


def get_async_llm() -> ChatOpenAI:
    """供 LangGraph Agent 使用的异步 LLM (LangChain 标准)"""
    return ChatOpenAI(
        model=_llm_cfg.llm_model,
        base_url=_llm_cfg.base_url,
        api_key=SecretStr(_llm_cfg.api_key),
        temperature=_llm_cfg.temperature,
        max_retries=_llm_cfg.max_retries,
        timeout=_llm_cfg.timeout,
        # default_headers=_get_private_headers(),
        # model_kwargs=_get_private_body_params(),
        http_client=http_client.sync_client,
        http_async_client=http_client.async_client,
    )


def get_async_vlm() -> ChatOpenAI:
    """供 LangGraph Agent 使用的异步 VLM (LangChain 标准)"""
    return ChatOpenAI(
        model=_llm_cfg.vlm_model,
        base_url=_llm_cfg.base_url,
        api_key=SecretStr(_llm_cfg.api_key),
        temperature=_llm_cfg.temperature,
        max_retries=_llm_cfg.max_retries,
        timeout=_llm_cfg.timeout,
        # default_headers=_get_private_headers(),
        # model_kwargs=_get_private_body_params(),
        http_client=http_client.sync_client,
        http_async_client=http_client.async_client,
    )


def get_llama_llm() -> OpenAILike:
    """供 LlamaIndex RAG 回答生成的 LLM (LlamaIndex 标准)"""

    return OpenAILike(
        model=_llm_cfg.llm_model,
        api_base=_llm_cfg.base_url,
        api_key=_llm_cfg.api_key,
        temperature=_llm_cfg.temperature,
        is_chat_model=True,
        default_headers=_get_private_headers(),
        additional_kwargs=_get_private_body_params(),
        max_retries=_llm_cfg.max_retries,
        timeout=_llm_cfg.timeout,
    )
