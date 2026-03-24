import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from demo.api.routers import review, system
from demo.config import get_settings
from demo.config.logger import setup_logging
from demo.llm.factory import debug_http_client
from demo.rag.core import setup_llama_index_env

setup_logging()
logger = logging.getLogger(__name__)
config = get_settings()


# ==========================================
# 🛡️ 定义生命周期管理器 (取代 on_event)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 【Startup 启动逻辑】
    logger.info(f"🚀 服务启动中... 环境: {config.app_env}")

    try:
        # 一键装配 LlamaIndex 全局 Settings
        setup_llama_index_env()
        logger.info("✅ LlamaIndex & RAG 引擎组件初始化成功")
    except Exception as e:
        logger.error(f"❌ 核心组件初始化失败: {e}", exc_info=True)

    yield  # 🌟 这里是应用运行的分割线

    # 2. 【Shutdown 关闭逻辑】
    logger.info("🛑 服务正在关闭，正在释放异步资源...")
    # 关闭全局异步客户端，防止连接泄露
    await debug_http_client.aclose()


# ==========================================
# 实例化 FastAPI 并挂载生命周期
# ==========================================
app = FastAPI(
    title="Modern AI Review API",
    description="企业级标书智能审核平台核心接口",
    lifespan=lifespan,
)

# 挂载路由
app.include_router(system.router)
app.include_router(review.router)
