# src/demo/rag/engine.py
from demo.config import get_settings
from demo.rag.pipeline import EnterpriseRAGPipeline


def get_company_rules_pipeline() -> EnterpriseRAGPipeline:
    """获取企业全局红线规章的检索流水线实例"""
    config = get_settings()
    return EnterpriseRAGPipeline(
        collection_name=config.milvus.collection_name,
        similarity_top_k=15,
        rerank_top_n=3,
        is_temporary=False,
    )
