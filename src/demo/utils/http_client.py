# src/demo/utils/http.py
import json
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional

import httpx

logger = logging.getLogger("demo")


class HttpClient:
    """
    企业级统一 HTTP 客户端管理器。
    支持全局连接池模式 (适合 FastAPI)，也支持临时隔离会话模式 (适合独立脚本)。
    """

    def __init__(self, timeout: float = 60.0):
        self._timeout = timeout
        # 企业级连接池配置：最大允许 100 个并发，保持 20 个长连接免握手
        self._limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)

        self._async_client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None

    # ==================== 1. 异步日志拦截器 ====================
    async def _async_log_request(self, request: httpx.Request) -> None:
        logger.debug(f"⬆️ [ASYNC REQ] {request.method} {request.url}")
        # 打印请求头 (过滤掉过长的或者敏感的 Token 可选)
        logger.debug(f"   [HEADERS]: {dict(request.headers)}")
        # 打印请求体
        if request.content:
            try:
                parsed = json.loads(request.content.decode("utf-8"))
                logger.debug(f"   [BODY]: {json.dumps(parsed, ensure_ascii=False)}")
            except Exception:
                logger.debug(
                    f"   [BODY RAW]: {request.content.decode('utf-8', errors='ignore')[:1000]}"
                )

    async def _async_log_response(self, response: httpx.Response) -> None:
        logger.debug(f"⬇️ [ASYNC RESP] Status: {response.status_code}")
        try:
            # 🌟 关键：调用 aread() 强行将流读入内存缓存，不影响业务层后续的 .json() 调用
            await response.aread()
            if response.content:
                try:
                    parsed = json.loads(response.content.decode("utf-8"))
                    logger.debug(f"   [BODY]: {json.dumps(parsed, ensure_ascii=False)}")
                except Exception:
                    logger.debug(
                        f"   [BODY RAW]: {response.content.decode('utf-8', errors='ignore')[:1000]}"
                    )
        except Exception as e:
            logger.debug(f"   [BODY READ ERROR]: 无法读取响应体 ({e})")

    # ==================== 2. 同步日志拦截器 ====================
    def _sync_log_request(self, request: httpx.Request) -> None:
        logger.debug(f"⬆️ [SYNC REQ] {request.method} {request.url}")
        logger.debug(f"   [HEADERS]: {dict(request.headers)}")
        if request.content:
            try:
                parsed = json.loads(request.content.decode("utf-8"))
                logger.debug(f"   [BODY]: {json.dumps(parsed, ensure_ascii=False)}")
            except Exception:
                logger.debug(
                    f"   [BODY RAW]: {request.content.decode('utf-8', errors='ignore')[:1000]}"
                )

    def _sync_log_response(self, response: httpx.Response) -> None:
        logger.debug(f"⬇️ [SYNC RESP] Status: {response.status_code}")
        try:
            # 🌟 关键：同步调用 read() 缓存数据
            response.read()
            if response.content:
                try:
                    parsed = json.loads(response.content.decode("utf-8"))
                    logger.debug(f"   [BODY]: {json.dumps(parsed, ensure_ascii=False)}")
                except Exception:
                    logger.debug(
                        f"   [BODY RAW]: {response.content.decode('utf-8', errors='ignore')[:1000]}"
                    )
        except Exception as e:
            logger.debug(f"   [BODY READ ERROR]: 无法读取响应体 ({e})")

    # ==================== 3. 内部客户端工厂 ====================
    def _create_async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout,
            limits=self._limits,
            event_hooks={
                "request": [self._async_log_request],
                "response": [self._async_log_response],
            },
        )

    def _create_sync_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self._timeout,
            limits=self._limits,
            event_hooks={
                "request": [self._sync_log_request],
                "response": [self._sync_log_response],
            },
        )

    # ==================== 4. 临时会话管理器 (即用即毁) ====================
    @asynccontextmanager
    async def async_session(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """
        获取一个临时的异步客户端，用完即毁。不依赖全局生命周期。
        用法: async with http_client.async_session() as client:
        """
        client = self._create_async_client()
        try:
            yield client
        finally:
            await client.aclose()

    @contextmanager
    def sync_session(self) -> Generator[httpx.Client, None, None]:
        """
        获取一个临时的同步客户端，用完即毁。不依赖全局生命周期。
        用法: with http_client.sync_session() as client:
        """
        client = self._create_sync_client()
        try:
            yield client
        finally:
            client.close()

    # ==================== 5. 全局生命周期 (连接池模式) ====================
    @property
    def async_client(self) -> httpx.AsyncClient:
        """获取全局异步连接池实例"""
        if self._async_client is None or self._async_client.is_closed:
            raise RuntimeError(
                "全局 AsyncClient 未初始化。请在 lifespan 中调用 await http_client.async_startup()"
            )
        return self._async_client

    async def async_startup(self, timeout: Optional[float] = None) -> None:
        """拉起全局异步连接池"""
        if timeout:
            self._timeout = timeout
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = self._create_async_client()
            logger.info("🌐 全局 AsyncClient 连接池初始化成功")

    async def async_shutdown(self) -> None:
        """释放全局异步连接池"""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
            logger.info("✅ 全局 AsyncClient 连接池已安全关闭")
        self._async_client = None

    @property
    def sync_client(self) -> httpx.Client:
        """获取全局同步连接池实例"""
        if self._sync_client is None or self._sync_client.is_closed:
            raise RuntimeError(
                "全局 SyncClient 未初始化。请先调用 http_client.sync_startup()"
            )
        return self._sync_client

    def sync_startup(self, timeout: Optional[float] = None) -> None:
        if timeout:
            self._timeout = timeout
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = self._create_sync_client()
            logger.info("🌐 全局 SyncClient 连接池初始化成功")

    def sync_shutdown(self) -> None:
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
            logger.info("✅ 全局 SyncClient 连接池已安全关闭")
        self._sync_client = None


# ==================== 全局单例导出 ====================
http_client = HttpClient()
