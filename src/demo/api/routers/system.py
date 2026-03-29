import logging
import shutil
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from langchain_core.messages import HumanMessage
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle
from pydantic import BaseModel

from demo.config.factory import get_async_llm
from demo.rag.engine import get_company_rules_pipeline
from demo.rag.ingestion import RAG_DOCS_DIT, build_knowledge_base

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["System & Knowledge"])

# 根据你的项目结构定位知识库存储目录
KB_DIR = RAG_DOCS_DIT


class StandardResponse(BaseModel):
    status: str
    message: str


@router.get("/health", response_model=StandardResponse)
async def health_check():
    return {"status": "success", "message": "Service is running gracefully."}


# ==========================================
# 1. 知识库上传与离线建库触发接口
# ==========================================
@router.post("/knowledge/upload", response_model=StandardResponse)
async def upload_and_ingest_knowledge(
    background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)
):
    """接收上传的制度文件，保存至 data 目录并触发向量库全量重建"""
    KB_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug(f"📁 知识库目录: {KB_DIR}")

    saved_files = []
    for file in files:
        if file.filename:
            file_path = KB_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)

    # build_knowledge_base 是原生 async 的
    # FastAPI 的 background_tasks 完美支持协程，直接传入即可，无需 to_thread
    background_tasks.add_task(build_knowledge_base)

    return {
        "status": "success",
        "message": f"成功保存 {len(saved_files)} 个文件。后台正在进行智能切片与 Milvus 混合建库...",
    }


# ==========================================
# 2. RAG 检索管道调试接口 (Debug)
# ==========================================
class DebugRequest(BaseModel):
    query: str


@router.post("/knowledge/debug")
async def debug_rag_retrieval(request: DebugRequest) -> dict[str, Any]:
    """
    接收测试 Query，返回大模型生成答案前：
    1. Milvus 混合检索 (Dense+BM25+RRF) 的初筛召回结果。
    2. 经过私有化 Reranker 重排过滤后的最终结果。
    """
    # 获取企业级流水线实例
    pipeline = get_company_rules_pipeline()
    query_bundle = QueryBundle(query_str=request.query)

    # 1. 执行第一阶段：异步混合初筛召回
    # (为了在 Debug 中拿到中间态，我们拆开调用流水线底层逻辑)
    vector_store = pipeline._get_vector_store(overwrite=False)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=pipeline.dense_embed_model
    )
    retriever = index.as_retriever(
        vector_store_query_mode="hybrid", similarity_top_k=pipeline.similarity_top_k
    )
    raw_nodes = await retriever.aretrieve(request.query)

    # 2. 执行第二阶段：私有化深度重排
    reranked_nodes = await pipeline.reranker.apostprocess_nodes(
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


# ==========================================
# 3. 智能问答接口 (Chat)
# ==========================================
class ChatQueryRequest(BaseModel):
    query: str
    use_rag: bool = True


class ChatQueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@router.post("/knowledge/chat", response_model=ChatQueryResponse)
async def chat_with_knowledge_base(request: ChatQueryRequest):
    """
    智能问答接口：支持自由切换 RAG 模式与纯大模型模式
    """
    try:
        # 统一使用带有强制并发排队保护机制的 LangChain LLM 实例
        llm = get_async_llm()

        if request.use_rag:
            # 🌟 架构升级：调用最新的企业级流水线一键查询
            pipeline = get_company_rules_pipeline()
            nodes = await pipeline.aretrieve(request.query)

            # 整理前端来源展示
            sources = [
                {
                    "text": n.node.get_content()[:200] + "...",
                    "score": round(n.score, 4) if n.score else None,
                    "file_name": n.node.metadata.get("file_name", "未知文件"),
                }
                for n in nodes
            ]

            # 统一生成逻辑：手搓高维 Prompt 投喂给 LangChain，代替原先黑盒的 aquery
            context_str = "\n\n---\n\n".join([n.node.get_content() for n in nodes])
            prompt = f"""你是一个严谨的企业规章制度助手。请基于以下检索到的【参考资料】回答问题。
要求：
1. 严格依据参考资料进行回答，绝不允许编造或凭借自身的训练数据发散。
2. 如果参考资料中没有相关信息，请明确回答“抱歉，在现有制度文件中未检索到相关规定”。

【参考资料】:
{context_str}

【用户问题】: {request.query}
"""
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return ChatQueryResponse(answer=str(response.content), sources=sources)

        else:
            # 脱离知识库，直接走纯大模型对话
            messages = [HumanMessage(content=request.query)]
            response = await llm.ainvoke(messages)

            return ChatQueryResponse(
                answer=str(response.content),
                sources=[],
            )

    except Exception as e:
        logger.error(f"❌ 问答生成发生严重异常: {e}", exc_info=True)
        return ChatQueryResponse(answer=f"系统降级处理，发生错误: {str(e)}", sources=[])
