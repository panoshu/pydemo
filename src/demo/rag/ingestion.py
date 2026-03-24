import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)

from demo.config import get_settings
from demo.rag.core import PROJECT_ROOT, get_milvus_store, setup_llama_index_env

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def get_chunking_strategy(config):
    strategy = config.rag.chunk_strategy
    if strategy == "sentence":
        return SentenceSplitter(
            chunk_size=config.rag.chunk_size, chunk_overlap=config.rag.chunk_overlap
        )
    elif strategy == "fixed_length":
        return TokenTextSplitter(
            chunk_size=config.rag.chunk_size, chunk_overlap=config.rag.chunk_overlap
        )
    elif strategy == "markdown":
        return MarkdownNodeParser()
    raise ValueError(f"未知的切片策略: {strategy}")


def build_knowledge_base(data_dir: Path | str = DEFAULT_DATA_DIR):
    """构建企业级向量知识库"""
    data_dir_path = Path(data_dir)
    logger.info(f"🚀 开始构建知识库，数据源: {data_dir_path}")

    config = get_settings()

    # 1. 注入全局环境变量 (LLM & Embedding)
    setup_llama_index_env()

    if not data_dir_path.exists():
        data_dir_path.mkdir(parents=True, exist_ok=True)
        with open(data_dir_path / "rule.txt", "w", encoding="utf-8") as f:
            f.write("合规红线：严禁在标书中承诺低于成本价的报价。\n")

    documents = SimpleDirectoryReader(str(data_dir_path)).load_data()
    if not documents:
        logger.warning("⚠️ 数据目录为空，未发现文档。")
        return

    # 1. 提取所有文档的纯文本，作为 BM25 的语料库
    corpus_texts = [doc.get_content() for doc in documents if doc.get_content()]

    # 2. 获取存储上下文 (⚠️ 这里使用 overwrite=True 重建集合，如果是追加请改为 False)
    vector_store = get_milvus_store(overwrite=True, corpus_texts=corpus_texts)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    splitter = get_chunking_strategy(config)

    # 3. 向量化并落盘 (因为 Settings 已接管全局 embed_model，无需再手动传入)
    logger.info("🧠 正在切片并自动持久化到 Milvus Lite...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    logger.info("✅ 知识库构建完毕！")


if __name__ == "__main__":
    build_knowledge_base()
