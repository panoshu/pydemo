import asyncio
import logging
import shutil
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from langchain_core.messages import HumanMessage
from llama_index.core.schema import QueryBundle
from pydantic import BaseModel

from demo.config import get_settings
from demo.llm.factory import get_async_llm, get_reranker
from demo.rag.engine import build_rag_engine
from demo.rag.ingestion import DEFAULT_DATA_DIR, build_knowledge_base

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["System & Knowledge"])


class StandardResponse(BaseModel):
    status: str
    message: str


@router.get("/health", response_model=StandardResponse)
async def health_check():
    return {"status": "success", "message": "Service is running gracefully."}


@router.post("/knowledge/upload", response_model=StandardResponse)
async def upload_and_ingest_knowledge(
    background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)
):
    """接收上传的制度文件，保存至 data 目录并触发向量库重构"""
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file in files:
        if file.filename:
            file_path = DEFAULT_DATA_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)

    # 依然放在后台任务执行，因为文件解析和 Embedding 调用较慢
    background_tasks.add_task(asyncio.to_thread, build_knowledge_base)

    return {
        "status": "success",
        "message": f"成功保存 {len(saved_files)} 个文件。后台正在进行切片与自动持久化至 Milvus...",
    }


# ==========================================
# RAG 检索管道调试接口
# ==========================================


class DebugRequest(BaseModel):
    query: str


@router.post("/knowledge/debug")
async def debug_rag_retrieval(request: DebugRequest) -> dict[str, Any]:
    """
    接收测试 Query，返回大模型生成答案前：
    1. Milvus 向量检索的初筛召回结果。
    2. 经过 Reranker 重排过滤后的最终结果。
    """
    config = get_settings()

    # 【修改点 2】：获取最新的引擎构建器
    query_engine = build_rag_engine()
    query_bundle = QueryBundle(query_str=request.query)

    # 1. 执行异步检索 (触发 Milvus 初筛)
    raw_nodes = await query_engine.retriever.aretrieve(request.query)

    # 【修改点 3】：通过工厂极简获取 Reranker，告别手动拼装配置！
    reranker = get_reranker(top_n=config.rag.similarity_top_k)
    reranked_nodes = raw_nodes

    if reranker:
        # 【修改点 4】：抛弃 asyncio.to_thread！直接调用 LlamaIndex 提供的原生异步重排钩子
        # 底层会自动路由到我们重写过的 _apostprocess_nodes
        reranked_nodes = await reranker.apostprocess_nodes(
            raw_nodes, query_bundle=query_bundle
        )

    # 格式化输出供前端渲染
    def format_nodes(nodes):
        return [
            {
                "score": round(float(n.score), 4) if n.score else 0.0,
                "text": n.node.get_content().strip(),
                "file_name": n.node.metadata.get("file_name", "未知来源"),
            }
            for n in nodes
        ]

    return {
        "status": "success",
        "raw_recall_count": len(raw_nodes),
        "reranked_count": len(reranked_nodes),
        "raw_nodes": format_nodes(raw_nodes),
        "reranked_nodes": format_nodes(reranked_nodes),
    }

# 定义接收问答请求的 Schema
class ChatQueryRequest(BaseModel):
    query: str
    use_rag: bool = True


# 定义返回结构的 Schema
class ChatQueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@router.post("/knowledge/chat", response_model=ChatQueryResponse)
async def chat_with_knowledge_base(request: ChatQueryRequest):
    """
    智能问答接口：支持自由切换 RAG 模式与纯大模型模式
    """
    try:
        if request.use_rag:
            # ==========================================
            # 模式 A：走 LlamaIndex RAG 检索链路
            # ==========================================
            query_engine = build_rag_engine()
            response = await query_engine.aquery(request.query)

            sources = []
            for node in response.source_nodes:
                sources.append(
                    {
                        "text": node.node.get_content()[:200] + "...",
                        "score": round(node.score, 4) if node.score else None,
                        "file_name": node.node.metadata.get("file_name", "未知文件"),
                    }
                )

            return ChatQueryResponse(answer=str(response), sources=sources)

        else:
            # ==========================================
            # 模式 B：脱离知识库，直接走 LangChain 纯大模型对话
            # ==========================================
            llm = get_async_llm()
            messages = [HumanMessage(content=request.query)]

            # 直接向私有化大模型发起对话请求
            response = await llm.ainvoke(messages)

            return ChatQueryResponse(
                answer=str(response.content),
                sources=[],  # 纯大模型没有外部参考来源
            )

    except Exception as e:
        logger.error(f"问答生成异常: {e}", exc_info=True)
        return ChatQueryResponse(answer=f"系统错误: {str(e)}", sources=[])
