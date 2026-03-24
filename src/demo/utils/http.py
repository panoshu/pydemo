import json
import logging

import httpx

logger = logging.getLogger(__name__)


async def log_request_info(request: httpx.Request):
    """拦截出站请求，打印完整 JSON Body"""
    logger.debug(f"{'=' * 20} ⬆️ [HTTP REQUEST] ⬆️ {'=' * 20}")
    logger.debug(f"URL: {request.method} {request.url}")
    # 隐藏敏感的 Authorization 或 API Key 头信息
    safe_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ["authorization", "api-key"]
    }
    logger.debug(f"Headers: {safe_headers}")

    if request.content:
        try:
            parsed = json.loads(request.content.decode("utf-8"))
            logger.debug(f"Body:\n{json.dumps(parsed, indent=2, ensure_ascii=False)}")
        except Exception:
            logger.debug(f"Body (Raw): {request.content.decode('utf-8')}")
    logger.debug("=" * 60)


async def log_response_info(response: httpx.Response):
    """拦截入站响应，打印完整 JSON Body"""
    logger.debug(f"{'=' * 20} ⬇️ [HTTP RESPONSE] ⬇️ {'=' * 20}")
    logger.debug(f"Status: {response.status_code}")

    await response.aread()
    try:
        parsed = json.loads(response.text)
        logger.debug(f"Body:\n{json.dumps(parsed, indent=2, ensure_ascii=False)}")
    except Exception:
        logger.debug(f"Body (Raw): {response.text}")
    logger.debug("=" * 60)


def get_debug_http_client(timeout: float = 120.0) -> httpx.AsyncClient:
    """
    获取一个注入了日志拦截器的标准异步 HTTP 客户端。
    可以在生产环境中通过判断 config.app_env 来决定是否挂载 event_hooks。
    """
    return httpx.AsyncClient(
        event_hooks={"request": [log_request_info], "response": [log_response_info]},
        timeout=timeout,
    )
