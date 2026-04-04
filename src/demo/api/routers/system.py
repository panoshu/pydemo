# src/demo/api/routers/system.py
import json
import logging
import shutil
from typing import Any, AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle
from pydantic import BaseModel

from demo.agents.chat import build_chat_graph
from demo.config.factory import get_async_llm
from demo.rag.engine import get_company_rules_pipeline
from demo.rag.ingestion import RAG_DOCS_DIT, build_knowledge_base

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["System & Knowledge"])

KB_DIR = RAG_DOCS_DIT


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
    KB_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug(f"📁 知识库目录: {KB_DIR}")

    saved_files = []
    for file in files:
        if file.filename:
            file_path = KB_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)

    background_tasks.add_task(build_knowledge_base)

    return {
        "status": "success",
        "message": f"成功保存 {len(saved_files)} 个文件。后台正在进行智能切片与 Milvus 混合建库...",
    }


class DebugRequest(BaseModel):
    query: str


@router.post("/knowledge/debug")
async def debug_rag_retrieval(request: DebugRequest) -> dict[str, Any]:
    pipeline = get_company_rules_pipeline()
    query_bundle = QueryBundle(query_str=request.query)

    vector_store = pipeline._get_vector_store(overwrite=False)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=pipeline.dense_embed_model
    )
    retriever = index.as_retriever(
        vector_store_query_mode="hybrid", similarity_top_k=pipeline.similarity_top_k
    )
    raw_nodes = await retriever.aretrieve(request.query)

    reranked_nodes = await pipeline.reranker.apostprocess_nodes(
        raw_nodes, query_bundle=query_bundle
    )

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


# ==========================================
# 3. 智能问答接口 (Chat) - 已重构为流式
# ==========================================
class ChatQueryRequest(BaseModel):
    query: str
    use_rag: bool = True
    session_id: str = "default_session"  # 👈 新增：接收前端传来的会话 ID


# 移除 response_model=ChatQueryResponse，因为我们改用 StreamingResponse
@router.post("/knowledge/chat")
async def chat_with_knowledge_base(request: ChatQueryRequest):
    """
    智能问答接口：支持自由切换 RAG 模式与纯大模型模式（流式输出）
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            llm = get_async_llm()

            if request.use_rag:
                pipeline = get_company_rules_pipeline()

                # 🌟 保持检索与重排为非流式：一次性查询并等待全部结果
                nodes = await pipeline.aretrieve(request.query)

                sources = [
                    {
                        "text": n.node.get_content()[:200] + "...",
                        "score": round(n.score, 4) if n.score else None,
                        "file_name": n.node.metadata.get("file_name", "未知文件"),
                    }
                    for n in nodes
                ]

                # 优先把召回的来源数据发给前端渲染
                yield json.dumps({"type": "sources", "data": sources}) + "\n"

                context_str = "\n\n---\n\n".join([n.node.get_content() for n in nodes])
                prompt = f"""你是一个严谨的企业规章制度助手。请基于以下检索到的【参考资料】回答问题。
要求：
1. 严格依据参考资料进行回答，绝不允许编造或凭借自身的训练数据发散。
2. 如果参考资料中没有相关信息，请明确回答“抱歉，在现有制度文件中未检索到相关规定”。

【参考资料】:
{context_str}

【用户问题】: {request.query}
"""
                # 🌟 仅将 LLM 推理过程改为流式生成
                async for chunk in llm.astream([HumanMessage(content=prompt)]):
                    yield json.dumps({"type": "chunk", "content": chunk.content}) + "\n"

            else:
                # 纯大模型模式
                yield json.dumps({"type": "sources", "data": []}) + "\n"
                messages = [HumanMessage(content=request.query)]
                async for chunk in llm.astream(messages):
                    yield json.dumps({"type": "chunk", "content": chunk.content}) + "\n"

        except Exception as e:
            logger.error(f"❌ 问答生成发生严重异常: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
