import logging
from typing import cast

from pydantic import BaseModel, Field

from demo.llm.factory import get_async_llm
from demo.state import ReviewState

logger = logging.getLogger(__name__)


# 定义结构化输出模型
class GrammarResult(BaseModel):
    has_error: bool = Field(description="是否存在语法错误、错别字或语气不当")
    errors: list[str] = Field(
        description="具体的错误列表描述，若无则为空数组", default_factory=list
    )


async def check_grammar_and_tone(state: ReviewState) -> dict:
    logger.info("🤖 [Agent: 语法审查] 异步审查开始...")
    try:
        # 获取异步 LLM 并绑定结构化输出
        llm = get_async_llm().with_structured_output(GrammarResult)
        prompt = f"请检查以下标书文本的语法、错别字和语气，语气必须严谨专业。\n\n文本：\n{state['document_text']}"

        # 【重构点】异步调用
        result = cast(GrammarResult, await llm.ainvoke(prompt))

        issues = []
        if result.has_error:
            for err in result.errors:
                issues.append({"category": "语法与语气", "message": err})

        return {"issues": issues}

    except Exception as e:
        logger.error(f"语法审查异常: {e}", exc_info=True)
        # 【重构点】容错：若大模型调用失败，不阻断流程，仅返回警告信息
        return {
            "issues": [
                {
                    "category": "系统警告 (语法审查)",
                    "message": "该节点处理超时或失败，暂未获得结果",
                }
            ]
        }
