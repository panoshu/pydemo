import asyncio
import base64
import logging
import os
import tempfile

from fastapi import APIRouter, BackgroundTasks

from demo.schemas.review import ReviewRequest, ReviewResponse
from demo.workflow import build_async_workflow

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["AI Review"])


def cleanup_temp_files(file_paths: list[str]):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"🧹 [清理] 已删除临时文件: {path}")
        except Exception as e:
            logger.error(f"清理文件 {path} 失败: {e}")


# 【重构点 1】改为 async def
@router.post("/review", response_model=ReviewResponse)
async def run_bid_review(request: ReviewRequest, background_tasks: BackgroundTasks):
    temp_image_paths = []

    # 【重构点 2】将耗时 IO (文件写操作) 放入线程池，防止阻塞 Async Event Loop
    def write_images():
        paths = []
        for img_b64 in request.images_base64:
            fd, path = tempfile.mkstemp(suffix=".jpg")
            with os.fdopen(fd, "wb") as f:
                f.write(base64.b64decode(img_b64))
            paths.append(path)
        return paths

    if request.images_base64:
        temp_image_paths = await asyncio.to_thread(write_images)

    background_tasks.add_task(cleanup_temp_files, temp_image_paths)

    initial_state = {
        "document_text": request.document_text,
        "image_paths": temp_image_paths,
        "issues": [],
        "extracted_data": {},
    }

    logger.info(
        f"🚀 开始并行异步执行工作流 (文档长度: {len(request.document_text)}, 图片数: {len(temp_image_paths)})..."
    )

    graph_app = build_async_workflow()

    # 【重构点 3】使用 ainvoke 异步执行 LangGraph，彻底释放工作线程
    final_state = await graph_app.ainvoke(initial_state)

    formatted_issues = [
        f"【{issue['category']}】: {issue['message']}"
        for issue in final_state.get("issues", [])
    ]

    return ReviewResponse(
        status="success",
        has_issues=len(formatted_issues) > 0,
        issues=formatted_issues,
        extracted_data=final_state.get("extracted_data", {}),
    )
