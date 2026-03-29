# src/demo/rag/pipeline.py
import asyncio
import logging
from typing import List

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.schema import Document, NodeWithScore, QueryBundle
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import connections, utility

from demo.config import get_settings
from demo.rag.components.embedding import PrivateAPIEmbedding
from demo.rag.components.reranker import PrivateAPIReranker
from demo.rag.components.sparse import (
    get_sparse_embedding_function,
)
from demo.rag.core import RAG_ROOT_DIR

logger = logging.getLogger(__name__)


class EnterpriseRAGPipeline:
    """
    企业级混合检索 RAG 流水线核心引擎。
    集成了 Milvus 服务端 BM25、私有大模型与并发安全的重排机制。
    """

    def __init__(
        self,
        collection_name: str,
        similarity_top_k: int = 10,
        rerank_top_n: int = 3,
        chunk_size: int = 512,
        is_temporary: bool = False,
    ):
        self.config = get_settings()
        self.collection_name = collection_name
        self.is_temporary = is_temporary
        self.chunk_size = chunk_size
        self.similarity_top_k = similarity_top_k

        self._db_path = RAG_ROOT_DIR / self.config.milvus.uri.lstrip("./")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # 显式实例化私有组件
        self.dense_embed_model = PrivateAPIEmbedding(embed_batch_size=10)
        self.reranker = PrivateAPIReranker(top_n=rerank_top_n)

        # 接入自定义 BM25 稀疏模型
        self.sparse_embed_model = get_sparse_embedding_function()

    def _get_vector_store(self, overwrite: bool = False) -> MilvusVectorStore:
        """获取挂载了双路检索引擎的 Milvus Store"""
        # 尝试挂载当前的事件循环，防止在异步上下文中初始化失败
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass

        return MilvusVectorStore(
            uri=str(self._db_path),
            collection_name=self.collection_name,
            dim=self.config.milvus.dim,
            similarity_metric=self.config.milvus.metric_type,
            overwrite=overwrite,
            enable_sparse=True,
            sparse_embedding_function=self.sparse_embed_model,
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 60},
        )

    # ==========================================
    # 数据注入层 (Ingestion)
    # ==========================================
    def ingest_directory(self, dir_path: str):
        """扫描本地目录并全量重建知识库 (通常用于离线建库脚本)"""
        logger.info(f"📂 正在扫描知识库目录: {dir_path}")
        documents = SimpleDirectoryReader(dir_path, recursive=True).load_data()
        if not documents:
            logger.warning("⚠️ 目录为空，跳过建库。")
            return
        self._ingest_documents(documents)

    def ingest_text(self, text: str) -> List[str]:
        """将单篇长文本切片并建库，返回切片原文列表 (用于临时标书库)"""
        logger.info("📄 正在处理超长文本切片入库...")
        return self._ingest_documents([Document(text=text)])

    def _get_chunking_strategy(self):
        strategy = self.config.rag.chunk_strategy
        if strategy == "sentence":
            return SentenceSplitter(
                chunk_size=self.config.rag.chunk_size,
                chunk_overlap=self.config.rag.chunk_overlap,
            )
        elif strategy == "fixed_length":
            return TokenTextSplitter(
                chunk_size=self.config.rag.chunk_size,
                chunk_overlap=self.config.rag.chunk_overlap,
            )
        elif strategy == "markdown":
            return MarkdownNodeParser()
        raise ValueError(f"未知的切片策略: {strategy}")

    def _ingest_documents(self, documents: List[Document]) -> List[str]:
        """核心切片与入库逻辑"""
        splitter = self._get_chunking_strategy()
        nodes = splitter.get_nodes_from_documents(documents)
        chunk_texts = [n.get_content() for n in nodes]

        vector_store = self._get_vector_store(overwrite=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.dense_embed_model,
            show_progress=False,
        )
        logger.info(
            f"✅ 知识库 [{self.collection_name}] 构建完毕，共 {len(nodes)} 个切片。"
        )
        return chunk_texts

    # ==========================================
    # 检索层 (Retrieval)
    # ==========================================
    async def aretrieve(self, query: str) -> List[NodeWithScore]:
        """异步执行：双路混合召回 -> 私有大模型重排"""
        vector_store = self._get_vector_store(overwrite=False)
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=self.dense_embed_model
        )

        # 必须显式开启 hybrid 模式，否则默认只查 Dense 向量
        retriever = index.as_retriever(
            vector_store_query_mode="hybrid", similarity_top_k=self.similarity_top_k
        )

        # 1. 底层 Milvus 并发执行 Dense + Sparse 召回并用 RRF 融合
        initial_nodes = await retriever.aretrieve(query)

        # 2. 将初筛节点送入私有 Reranker 进行深度语义打分
        query_bundle = QueryBundle(query_str=query)
        final_nodes = await self.reranker.apostprocess_nodes(
            initial_nodes, query_bundle
        )

        return final_nodes

    # ==========================================
    # 生命周期层 (Lifecycle)
    # ==========================================
    def destroy(self):
        """物理销毁表结构与数据 (仅对临时库生效)"""
        if self.is_temporary:
            logger.info(f"🧹 正在销毁临时向量库: {self.collection_name}...")
            try:
                connections.connect("default", uri=self.config.milvus.uri)
                if utility.has_collection(self.collection_name):
                    utility.drop_collection(self.collection_name)  # type: ignore
                    logger.debug(f"🗑️ 表 {self.collection_name} 已安全释放。")
            except Exception as e:
                logger.error(f"❌ 销毁临时库失败: {e}")
