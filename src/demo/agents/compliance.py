import logging

from demo.rag.engine import build_rag_engine
from demo.state import ReviewState

logger = logging.getLogger(__name__)


async def check_compliance_with_rag(state: ReviewState) -> dict:
    logger.info("🤖 [Agent: RAG合规] 异步审查开始...")
    try:
        query_engine = build_rag_engine()

        query_str = (
            "请作为合规审查专家，严格对比知识库中的红线与规定，检查以下标书内容是否违规。\n"
            "如果有违规，请明确指出违规点；如果完全符合，请回复'未发现违规'。\n"
            f"标书内容：\n{state['document_text']}"
        )

        # 【重构点】必须使用 aquery 进行异步检索与生成
        response = await query_engine.aquery(query_str)
        res_text = str(response).strip()

        issues = []
        # 简单的规则判断，真实企业场景可引入第二次 LLM 调用判断是否为违规
        if "未发现" not in res_text and "不违反" not in res_text:
            issues.append({"category": "合规红线违规", "message": res_text})

        return {"issues": issues}

    except Exception as e:
        logger.error(f"RAG合规审查异常: {e}", exc_info=True)
        return {
            "issues": [
                {
                    "category": "系统警告 (RAG合规审查)",
                    "message": "该节点知识库检索超时或失败，暂未获得结果",
                }
            ]
        }
