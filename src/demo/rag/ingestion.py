# src/demo/rag/ingestion.py
import asyncio
import logging

from demo.config.logger import setup_logging
from demo.rag.core import RAG_ROOT_DIR, setup_llama_index_env
from demo.rag.engine import get_company_rules_pipeline

logger = logging.getLogger("demo")


RAG_DOCS_DIT = RAG_ROOT_DIR / "docs"


async def build_knowledge_base():
    """全量构建企业规章制度知识库"""
    logger.info("🚀 启动企业级规章知识库构建流程...")

    setup_llama_index_env()

    kb_dir = RAG_DOCS_DIT
    if not kb_dir.exists():
        logger.error(f"❌ 知识库目录不存在: {kb_dir}")
        kb_dir.mkdir(parents=True, exist_ok=True)
        logger.info("💡 已自动创建目录，请放入 txt/pdf 等企业规章文件后再次运行。")
        return

    pipeline = get_company_rules_pipeline()
    # 底层会自动覆写 collection，完成 Dense 和内置 BM25 数据的双重注入
    pipeline.ingest_directory(str(kb_dir))

    logger.info("🎉 知识库全量构建完毕！混合检索索引已就绪。")


if __name__ == "__main__":
    setup_logging()
    asyncio.run(build_knowledge_base())
