# src/demo/agents/grammar.py
import asyncio
import logging
from typing import TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage

from demo.config.factory import get_async_llm
from demo.schemas.review import GrammarAgentResult

logger = logging.getLogger(__name__)


# ==========================================
# 1. 动态子节点状态定义
# ==========================================
class ChunkState(TypedDict):
    """Map-Reduce 中每个并行节点独立持有的子状态"""

    chunk_index: int
    chunk_text: str


# ==========================================
# 2. 全局并发控制器 (防止打挂私有大模型 API)
# ==========================================
# 🛡️ 严格设置为 1，保障测试环境与私有算力环境的绝对稳定排队
LLM_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(1)


# ==========================================
# 3. Worker 节点：真正执行单片段审查的逻辑
# ==========================================
async def process_single_chunk(state: ChunkState) -> dict:
    """
    【Worker 节点】：处理单一的文本片段，寻找语病与逻辑漏洞
    """
    chunk_text = state["chunk_text"]
    chunk_index = state["chunk_index"]

    system_prompt = """你是一位严谨的企业标书审查专家（文字与逻辑校对方向）。
请阅读给定的标书文本片段，找出其中存在的：
1. 明显的语法错误或错别字（如把“我公司”写成“我功司”）。
2. 严重的上下文逻辑断裂或表达不清。

如果该片段没有任何问题，请直接返回 has_issues=False。
如果发现问题，请详细记录问题分类、描述以及原文证据。"""

    # 获取大模型并绑定强制输出结构
    llm = get_async_llm().with_structured_output(GrammarAgentResult)

    try:
        # 🔑 核心防护：获取信号量后再发起网络请求，严格控制并发排队
        async with LLM_CONCURRENCY_SEMAPHORE:
            raw_result = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"【标书片段内容】：\n{chunk_text}"),
                ]
            )
            result = cast(GrammarAgentResult, raw_result)

        # 🛡️ 核心防御：拦截大模型 API 资源不足导致的“伪 200” NoneType 异常
        if result is None:
            logger.error(
                f"❌ 大模型返回 NoneType (服务器资源不足或解析失败)。片段索引: {chunk_index}"
            )
            return {
                "issues": [
                    {
                        "category": "系统降级",
                        "message": f"片段 {chunk_index} 审查时服务器资源不足，已跳过。",
                        "evidence": None,
                        "chunk_index": chunk_index,
                    }
                ]
            }

        # 如果大模型没有发现问题，返回空列表
        if not result.has_issues or not result.issues:
            return {"issues": []}

        # 将结构化的数据转为字典格式，供 LangGraph 的 operator.add 安全追加汇总
        formatted_issues = []
        for issue in result.issues:
            formatted_issues.append(
                {
                    "category": f"通读审查 ({issue.category})",
                    "message": issue.message,
                    "evidence": issue.evidence,
                    "chunk_index": chunk_index,  # 记录是在哪个片段发现的，方便后续溯源
                }
            )

        return {"issues": formatted_issues}

    except Exception as e:
        logger.error(f"❌ 审查片段 {chunk_index} 失败: {e}", exc_info=True)
        # 容错处理：单点故障绝不中断整个有向无环图
        return {"error_traces": [f"片段 {chunk_index} 审查异常: {str(e)}"]}
