from typing import Dict, Optional

from langchain_openai import ChatOpenAI
from llama_index.llms.openai_like import OpenAILike as LlamaOpenAI
from pydantic import SecretStr

from demo.config import get_settings
from demo.llm.embedding import PrivateAPIEmbedding
from demo.rag.reranker import PrivateAPIReranker
from demo.utils.http import get_debug_http_client


def _get_private_headers() -> Dict[str, str]:
    """
    【基建核心】统一组装私有化 API 所需的自定义请求头。
    将配置中的 app_token, biz_no 等转化为真实的 HTTP Headers。
    """
    config = get_settings()
    private_ext = getattr(config.llm, "provider_private", None)

    headers = {}
    if private_ext:
        headers.update(
            {
                "app-token": private_ext.app_token,
            }
        )
    return headers


def _get_private_body_params() -> dict:
    """
    【新增核心基建】：获取需要透传到 JSON 请求体（Body）里的私有参数
    """
    config = get_settings()
    private_ext = getattr(config.llm, "provider_private", None)

    params = {}
    if private_ext:
        if private_ext.biz_no:
            params["bizNo"] = private_ext.biz_no

    # 如果还需要强制透传 do_sample 等参数，也可以在这里加
    # params["do_sample"] = False
    return params


def get_async_llm() -> ChatOpenAI:
    config = get_settings()
    llm_cfg = config.llm

    _headers = _get_private_headers()
    _body_params = _get_private_body_params()

    return ChatOpenAI(
        model=llm_cfg.llm_model,
        base_url=llm_cfg.base_url,
        api_key=SecretStr(llm_cfg.api_key),
        temperature=llm_cfg.temperature,
        max_retries=3,
        http_async_client=get_debug_http_client(timeout=llm_cfg.timeout),
        timeout=llm_cfg.timeout,
        # default_headers=_headers,
        # model_kwargs=_body_params,
    )


def get_async_vlm() -> ChatOpenAI:
    config = get_settings()
    llm_cfg = config.llm

    _headers = _get_private_headers()
    _body_params = _get_private_body_params()

    return ChatOpenAI(
        model=llm_cfg.vlm_model,
        base_url=llm_cfg.base_url,
        api_key=SecretStr(llm_cfg.api_key),
        temperature=llm_cfg.temperature,
        max_retries=3,
        http_async_client=get_debug_http_client(timeout=llm_cfg.timeout),
        timeout=llm_cfg.timeout,
        # default_headers=_headers,
        # model_kwargs=_body_params,
    )


def get_llama_components() -> dict:
    config = get_settings()
    llm_cfg = config.llm

    _headers = _get_private_headers()
    _body_params = _get_private_body_params()

    llm = LlamaOpenAI(
        model=llm_cfg.llm_model,
        api_base=llm_cfg.base_url,
        api_key=llm_cfg.api_key,
        temperature=llm_cfg.temperature,
        context_window=llm_cfg.context_window,
        is_chat_model=True,
        timeout=llm_cfg.timeout,
        # default_headers=_headers,
        # additional_kwargs=_body_params,
    )

    # 提取私有化配置（如果存在）
    private_ext = llm_cfg.provider_private

    embed_model = PrivateAPIEmbedding.from_defaults(
        private_ext=private_ext, embed_batch_size=10
    )

    return {"llm": llm, "embed_model": embed_model}

def get_reranker(top_n: Optional[int] = None) -> Optional[PrivateAPIReranker]:
    """
    提供给 engine.py 或 system.py 测试用的快捷工厂方法
    """
    private_ext = get_settings().llm.provider_private
    return PrivateAPIReranker.from_defaults(private_ext=private_ext, top_n=top_n)
