import asyncio
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import jieba
from llama_index.core import Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction
from pymilvus.model.sparse import BM25EmbeddingFunction

from demo.config import get_settings
from demo.llm.factory import get_llama_components

logger = logging.getLogger(__name__)

# 动态计算项目根目录 (pydemo)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ==========================================
# 适配器：桥接 PyMilvus BM25 与 LlamaIndex
# ==========================================
class LocalBM25SparseEmbedding(BaseSparseEmbeddingFunction):
    """适配 LlamaIndex 接口标准的本地 BM25 稀疏向量生成器"""

    def __init__(self, idf_path: Path):
        self.bm25_ef = BM25EmbeddingFunction(analyzer=jieba.lcut)  # type: ignore
        self.idf_path = idf_path

    def train_and_save(self, corpus_texts: list[str]):
        """训练词频字典并持久化"""
        self.bm25_ef.fit(corpus_texts)
        self.bm25_ef.save(str(self.idf_path))

    def load_dictionary(self):
        """加载已有的词频字典"""
        if self.idf_path.exists():
            self.bm25_ef.load(str(self.idf_path))
        else:
            logger.warning(
                "⚠️ 未找到 BM25 词频字典！如果是查询阶段，稀疏检索将报错；如果是建库阶段请忽略。"
            )

    def _csr_to_dict_list(self, csr_matrix) -> list[dict[int, float]]:
        """【核心】将 scipy 稀疏矩阵转换为 LlamaIndex 要求的字典列表"""
        result = []
        for i in range(csr_matrix.shape[0]):
            row = csr_matrix.getrow(i)
            # 提取稀疏矩阵中非零的索引(词ID)和值(权重)
            result.append({int(k): float(v) for k, v in zip(row.indices, row.data)})
        return result

    def encode_queries(self, queries: list[str]) -> list[dict[int, float]]:
        """实现 LlamaIndex 的 Query 编码接口"""
        csr_matrix = self.bm25_ef.encode_queries(queries)
        return self._csr_to_dict_list(csr_matrix)

    def encode_documents(self, documents: list[str]) -> list[dict[int, float]]:
        """实现 LlamaIndex 的 Document 编码接口"""
        csr_matrix = self.bm25_ef.encode_documents(documents)
        return self._csr_to_dict_list(csr_matrix)


def setup_llama_index_env():
    """
    【基建函数 1】彻底接管 LlamaIndex 的全局运行环境。
    杜绝底层组件隐式回退到 OpenAI 导致报错或数据泄露。
    """
    logger.debug("⚙️ 正在初始化 LlamaIndex 全局 Settings...")
    components = get_llama_components()
    Settings.llm = components["llm"]
    Settings.embed_model = components["embed_model"]
    # 如果未来有全局 CallbackManager（如 Langfuse 监控），也在这里注册


def get_milvus_store(
    overwrite: bool = False, corpus_texts: list[str] | None = None
) -> MilvusVectorStore:
    """
    【基建函数 2】获取 Milvus Lite 向量数据库实例。
    统一处理绝对路径解析和目录创建逻辑，杜绝 SQLite 找不到目录的报错。

    :param overwrite: 是否覆盖已存在的 Collection。
            Ingestion (重建知识库) 时可设为 True，Engine 查询时必须为 False！
    """
    config = get_settings()
    db_path = PROJECT_ROOT / config.milvus.uri.lstrip("./")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"🔌 连接 Milvus Lite: {db_path} (overwrite={overwrite})")

    idf_path = db_path.parent / "bm25_idf.json"
    sparse_ef = LocalBM25SparseEmbedding(idf_path)

    # 2. 处理 BM25 的词频字典状态
    if overwrite and corpus_texts:
        logger.info("📊 正在全量扫描知识库，计算 BM25 IDF 词频字典...")
        sparse_ef.train_and_save(corpus_texts)
    else:
        sparse_ef.load_dictionary()

    def _init_store():
        return MilvusVectorStore(
            uri=str(db_path),
            collection_name=config.milvus.collection_name,
            dim=config.milvus.dim,
            similarity_metric=config.milvus.metric_type,
            overwrite=overwrite,
            enable_sparse=True,
            sparse_embedding_function=sparse_ef,
            hybrid_ranker="RRFReanker",
            hybrid_ranker_params={"k": 60},
        )

    try:
        asyncio.get_running_loop()
        return _init_store()
    except RuntimeError:
        logger.debug("🛡️ 正在注入临时 Running Loop...")

        async def _dummy_runner():
            return _init_store()

        return asyncio.run(_dummy_runner())
