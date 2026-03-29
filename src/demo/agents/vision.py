import base64
import logging
import os

from langchain_core.messages import HumanMessage

from demo.config.factory import get_async_vlm
from demo.state import ReviewState

logger = logging.getLogger(__name__)


def _encode_image(image_path: str) -> str:
    """将本地图片转为Base64以供VLM读取"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def check_images(state: ReviewState) -> dict:
    logger.info("🤖 [Agent: 视觉审查] 异步审查开始...")
    try:
        image_paths = state.get("image_paths", [])
        if not image_paths:
            logger.debug("无图片，跳过视觉审查。")
            return {"issues": []}

        vlm = get_async_vlm()
        issues = []

        # 遍历所有图片进行异步检查
        for path in image_paths:
            if not os.path.exists(path):
                continue

            base64_img = _encode_image(path)
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "请检查这张标书资质图片。是否存在未盖公章、模糊不清、疑似PS篡改等问题？如果存在请简要指出，如果完全正常请严格回复'正常'。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                    },
                ]
            )

            # 【重构点】异步调用 VLM
            response = await vlm.ainvoke([message])
            res_text = str(response.content).strip()

            if "正常" not in res_text:
                issues.append(
                    {
                        "category": "视觉与资质图片违规",
                        "message": f"图片异常 ({os.path.basename(path)}): {res_text}",
                    }
                )

        return {"issues": issues}

    except Exception as e:
        logger.error(f"视觉审查异常: {e}", exc_info=True)
        return {
            "issues": [
                {
                    "category": "系统警告 (视觉审查)",
                    "message": "多模态大模型处理失败，暂未获得图片审查结果",
                }
            ]
        }
