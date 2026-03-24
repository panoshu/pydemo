import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine

from demo.config import get_settings
from demo.llm.factory import get_reranker
from demo.rag.core import get_milvus_store, setup_llama_index_env

logger = logging.getLogger(__name__)


def build_rag_engine() -> RetrieverQueryEngine:
    """
    构造纯异步的 RAG 检索引擎。
    """
    logger.info("🔧 正在装配企业级 RAG 引擎 (Milvus + 异步 API Reranker)...")

    config = get_settings()

    # 1. 初始化全局 LLM 环境
    setup_llama_index_env()

    # 2. 连接 Milvus 库 (⚠️ 查询时严格禁止 overwrite，必须为 False)
    try:
        vector_store = get_milvus_store(overwrite=False)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    except Exception as e:
        logger.error(f"❌ 连接向量库失败: {e}")
        raise RuntimeError("请先运行 ingestion.py 构建知识库") from e

    # 3. 构建检索器
    recall_k = config.rag.similarity_top_k * config.rag.recall_multiplier
    retriever = index.as_retriever(similarity_top_k=recall_k)

    # 4. 挂载异步重排器 (假定你在 LlmConfig 中使用了 provider_private)
    node_postprocessors = []
    reranker = get_reranker(top_n=config.rag.similarity_top_k)
    if reranker:
        node_postprocessors.append(reranker)

    # 5. 组装引擎 (由于 Settings.llm 已生效，无需手动传入 llm 参数)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever, node_postprocessors=node_postprocessors
    )

    logger.info("✅ 异步 RAG 引擎装配完毕。")
    return query_engine
