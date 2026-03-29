import logging
from typing import cast

from pydantic import BaseModel, Field

from demo.config.factory import get_async_llm
from demo.state import ReviewIssue, ReviewState

logger = logging.getLogger(__name__)


class BidCoreInfo(BaseModel):
    response_time_hours: float = Field(
        description="服务响应时间(小时)。未提及则返回 999。"
    )


async def extract_and_verify_logic(state: ReviewState) -> dict:
    logger.info("🤖 [Agent: 逻辑结构] 异步审查开始...")
    try:
        llm = get_async_llm().with_structured_output(BidCoreInfo)
        prompt = f"提取标书中的服务响应时间：\n{state['document_text']}"

        # 【重构点】必须使用 ainvoke
        extracted = cast(BidCoreInfo, await llm.ainvoke(prompt))

        issues: list[ReviewIssue] = []
        if extracted.response_time_hours > 2 and extracted.response_time_hours != 999:
            issues.append(
                {
                    "category": "逻辑违规",
                    "message": f"承诺响应时间为 {extracted.response_time_hours} 小时，超2小时红线！",
                }
            )

        return {
            "extracted_data": {"response_time": extracted.response_time_hours},
            "issues": issues,
        }

    except Exception as e:
        logger.error(f"逻辑抽取异常: {e}", exc_info=True)
        # 【重构点】优雅降级，防止整个工作流崩溃
        return {
            "issues": [
                {
                    "category": "系统警告 (逻辑审查)",
                    "message": "该节点处理超时或失败，暂未获得结果",
                }
            ]
        }
