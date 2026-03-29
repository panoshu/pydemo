# src/demo/rag/temp_engine.py
import logging
from contextlib import asynccontextmanager

from demo.rag.pipeline import EnterpriseRAGPipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def temporary_bid_rag(document_text: str, session_id: str):
    """
    【企业级】基于生命周期管理的临时会话级 RAG 引擎
    使用 contextmanager 确保用完物理销毁，绝不污染全局数据库。
    """
    temp_collection = f"temp_bid_{session_id}"

    # 初始化临时流水线
    pipeline = EnterpriseRAGPipeline(
        collection_name=temp_collection,
    )

    try:
        logger.info(f"📦 [Session: {session_id}] 正在拉起临时标书 RAG 空间...")
        # 灌入数据并提取供并发 Map-Reduce 使用的切片
        chunk_texts = pipeline.ingest_text(document_text)

        # 将 pipeline 引擎暴露给 LangGraph 使用
        yield pipeline, chunk_texts

    finally:
        # 引擎内部处理物理销毁逻辑
        pipeline.destroy()
