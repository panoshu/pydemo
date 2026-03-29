# src/demo/api/routers/review.py
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from numpy import isin

from demo.rag.temp_engine import temporary_bid_rag
from demo.schemas.review import ReviewRequest
from demo.workflow import ReviewState, build_review_graph

router = APIRouter(prefix="/api/review", tags=["Review"])
logger = logging.getLogger(__name__)


@router.post("/stream")
async def stream_review(request: ReviewRequest):
    """
    企业级多智能体审查流式网关
    将底层 LangGraph 和 LlamaIndex 的作业进度通过 SSE 实时推送到前端
    """
    # 1. 生成全局唯一会话 ID，贯穿图网络与临时数据库
    session_id = uuid.uuid4().hex[:8]
    logger.info(f"🆕 收到审查请求，分配 Session: {session_id}")

    async def event_generator() -> AsyncGenerator[str, None]:
        # 初始握手
        yield (
            json.dumps({"status": "start", "message": "🚀 正在初始化审查流水线..."})
            + "\n"
        )

        try:
            # ==========================================
            # 2. 挂载临时双路 RAG 引擎 (上下文管理器确保绝对销毁)
            # ==========================================
            yield (
                json.dumps(
                    {
                        "status": "processing",
                        "agent": "系统编排",
                        "message": "📦 正在对标书进行切片并构建临时向量空间...",
                    }
                )
                + "\n"
            )

            async with temporary_bid_rag(request.document_text, session_id) as (
                bid_engine,
                chunks,
            ):
                yield (
                    json.dumps(
                        {
                            "status": "processing",
                            "agent": "系统编排",
                            "message": f"✅ 临时知识库建库完成，共产生 {len(chunks)} 个切片。正在唤醒智能体...",
                        }
                    )
                    + "\n"
                )

                # ==========================================
                # 3. 组装 LangGraph 全局初始状态
                # ==========================================
                initial_state: ReviewState = {
                    "document_text": request.document_text,
                    "images_base64": request.images_base64,
                    "checklist": request.checklist,
                    "enable_full_text_check": request.enable_full_text_check,
                    "enable_double_rag_check": request.enable_double_rag_check,
                    "document_chunks": chunks,
                    "session_id": session_id,
                    "issues": [],
                    "double_rag_logs": [],
                    "error_traces": [],
                    "extracted_data": {},
                }

                # ==========================================
                # 4. 执行图网络并捕获流式事件 (stream_mode="updates")
                # ==========================================
                app = build_review_graph()
                final_issues = []

                total_chunks = len(chunks)
                processed_chunks = 0

                graph_config: RunnableConfig = {
                    "max_concurrency": 1,
                }

                async for output in app.astream(
                    initial_state,
                    stream_mode="updates",
                    config=graph_config,
                ):
                    for node_name, node_update in output.items():
                        if isinstance(node_update, dict) and node_update.get("issues"):
                            final_issues.extend(node_update["issues"])

                        # 进度推送逻辑优化
                        if node_name == "dispatch":
                            yield (
                                json.dumps(
                                    {
                                        "status": "processing",
                                        "agent": "系统编排",
                                        "message": "🚦 任务已分发，通读与比对智能体正在排队串行执行...",
                                    }
                                )
                                + "\n"
                            )

                        elif node_name == "process_chunk_node":
                            # 累加进度并推送到前端
                            processed_chunks += 1
                            yield (
                                json.dumps(
                                    {
                                        "status": "processing",
                                        "agent": "📝 全文通读",
                                        "message": f"正在逐段精读标书... (进度: {processed_chunks}/{total_chunks})",
                                    }
                                )
                                + "\n"
                            )

                        elif node_name == "compliance_node":
                            yield (
                                json.dumps(
                                    {
                                        "status": "processing",
                                        "agent": "⚖️ 双向 RAG",
                                        "message": "完成基于 CheckList 的公司规章双边核对！",
                                    }
                                )
                                + "\n"
                            )

                # ==========================================
                # 5. 图网络执行完毕，组装最终结果
                # ==========================================
                result_data = {
                    "has_issues": len(final_issues) > 0,
                    "issues": final_issues,
                }
                yield json.dumps({"status": "final", "data": result_data}) + "\n"

        except Exception as e:
            logger.error(
                f"审查流水线严重异常 [Session: {session_id}]: {e}", exc_info=True
            )
            yield (
                json.dumps({"status": "error", "message": f"流水线中断: {str(e)}"})
                + "\n"
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")
