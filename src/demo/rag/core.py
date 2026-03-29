# src/demo/rag/core.py
import logging

from llama_index.core import Settings

from demo.config import PROJECT_ROOT, get_settings
from demo.config.factory import get_llama_llm
from demo.rag.components.embedding import PrivateAPIEmbedding

logger = logging.getLogger(__name__)

RAG_ROOT_DIR = PROJECT_ROOT / "data"


def setup_llama_index_env():
    """一键装配 LlamaIndex 全局环境"""
    logger.debug("⚙️ 正在初始化 LlamaIndex 全局 Settings...")

    config = get_settings()

    # 1. 文本生成 LLM 依然从 factory 获取
    Settings.llm = get_llama_llm()

    # 2. 🌟 向量化模型直接使用我们刚刚封装好的私有组件
    Settings.embed_model = PrivateAPIEmbedding(embed_batch_size=10)

    # 3. 默认切片策略
    Settings.chunk_size = config.rag.chunk_size
    Settings.chunk_overlap = config.rag.chunk_overlap
