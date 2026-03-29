# src/demo/rag/components/sparse.py
import logging
from typing import cast

from llama_index.vector_stores.milvus.utils import (
    BaseSparseEmbeddingFunction,
    BM25BuiltInFunction,
)

logger = logging.getLogger(__name__)

_analyzer_params = {
    "tokenizer": "jieba",
    "filter": [
        "lowercase",  # 1. 统一英文大小写
        "cnalphanumonly",  # 2. 去除所有中英文标点和特殊符号
        {
            "type": "stop",  # 3. 动态业务停用词字典
            "stop_words": [
                "的",
                "了",
                "和",
                "与",
                "及",
                "在",
                "对",
                "是",
                "将",
                "应",
                "项目",
                "公司",
                "招标",
                "投标",
                "规定",
                "要求",
                "相关",
                "文件",
                "附件",
                "甲方",
                "乙方",
                "采购",
                "服务",
                "工作",
                "进行",
                "提供",
                "必须",
            ],
        },
        {"type": "length", "max": 30},  # 4. 长度熔断防乱码
    ],
}


def get_sparse_embedding_function() -> BaseSparseEmbeddingFunction:
    """
    获取企业级深度定制的 Milvus 服务端 BM25 稀疏嵌入函数。
    配置了 Jieba 分词器、去标点符号以及动态停用词表。
    """

    return cast(
        BaseSparseEmbeddingFunction,
        BM25BuiltInFunction(
            analyzer_params=_analyzer_params,
            enable_match=True,
        ),
    )


if __name__ == "__main__":
    from pymilvus import (
        MilvusClient,
    )

    from demo.config import get_settings
    from demo.rag.core import RAG_ROOT_DIR

    config = get_settings()
    db_path = RAG_ROOT_DIR / config.milvus.uri.lstrip("./")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    client = MilvusClient(uri=config.milvus.uri, token="root:Milvus")

    # Sample text to analyze
    sample_text = "The Milvus vector database is built for scale!"

    # Run the standard analyzer with the defined configuration
    result = client.run_analyzer(sample_text, _analyzer_params)
    print("Standard analyzer output:", result)
