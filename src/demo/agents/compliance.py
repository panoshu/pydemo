# src/demo/agents/compliance.py
import logging
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from demo.config.factory import get_async_llm
from demo.rag.engine import get_company_rules_pipeline
from demo.rag.pipeline import EnterpriseRAGPipeline
from demo.schemas.review import ComplianceAgentResult
from demo.state import ReviewState

logger = logging.getLogger(__name__)


async def compliance_double_rag_node(state: ReviewState) -> dict:
    """
    【双向 RAG 智能体】：基于 Checklist 的双边检索与比对
    """
    if not state.get("enable_double_rag_check", True):
        logger.info("⏭️ 双向 RAG 审查策略已关闭，跳过此阶段。")
        return {}

    checklist = state.get("checklist", [])
    session_id = state.get("session_id")

    if not checklist or not session_id:
        logger.warning("⚠️ 缺失 checklist 或 session_id，无法执行双向 RAG。")
        return {}

    logger.info(
        f"⚖️ [Session: {session_id}] 启动双向 RAG 审查，共 {len(checklist)} 个核查项..."
    )

    # 🌟 1. 实例化两个极其强大的企业级混合检索流水线
    company_pipeline = get_company_rules_pipeline()
    bid_pipeline = EnterpriseRAGPipeline(
        collection_name=f"temp_bid_{session_id}",
    )

    # 2. 绑定带有结构化输出能力的大模型
    llm = get_async_llm().with_structured_output(ComplianceAgentResult)

    issues_found = []
    process_logs = []

    # 3. 遍历 Checklist 进行双向核对
    for check_item in checklist:
        logger.debug(f"🔍 正在双向核对清单项: 【{check_item}】")

        try:
            # ➡️ 动作 A：查阅公司红线 (一键调用，内置 Dense+BM25+Rerank)
            rule_nodes = await company_pipeline.aretrieve(
                f"查询【{check_item}】的底线规定或红线要求"
            )
            company_rules_context = (
                "\n".join([n.node.get_content() for n in rule_nodes])
                or "未检索到公司相关明确规章。"
            )

            # ➡️ 动作 B：查阅当前标书
            bid_nodes = await bid_pipeline.aretrieve(
                f"提取标书中关于【{check_item}】的具体描述或承诺数值"
            )
            bid_context = (
                "\n".join([n.node.get_content() for n in bid_nodes])
                or "标书中未提及此项内容。"
            )

            # ➡️ 动作 C：大模型化身裁判进行比对仲裁
            system_prompt = """你是一位铁面无私的企业招投标合规审计官。
你的任务是：比对【公司红线规章】与【标书实际条款】，判断标书是否违规。

规则：
1. 如果标书内容突破了公司红线底线，判定为不合规。
2. 如果公司规章明确要求必须有某项内容，但标书中未提及，判定为不合规。
3. 其他情况判定为合规。
请严格输出 JSON 格式的比对结果。"""

            human_prompt = f"""
=== 正在核对的清单项 ===
【{check_item}】

=== 公司红线规章 ===
{company_rules_context}

=== 标书实际条款 ===
{bid_context}

请执行严格对比并输出审查结果。
"""
            # 发起 LLM 仲裁
            raw_result = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]
            )
            result = cast(ComplianceAgentResult, raw_result)

            # 🛡️ 核心防御：拦截大模型 API 资源不足导致的“伪 200” NoneType 异常
            if result is None:
                logger.error(
                    f"❌ 大模型返回了空数据 (服务器资源可能不足)。核对项: {check_item}"
                )
                issues_found.append(
                    {
                        "category": f"系统降级 ({check_item})",
                        "message": "大模型服务资源不足或响应解析失败，跳过此项深度比对。",
                        "evidence": None,
                        "reference_rule": None,
                    }
                )
                continue

            # ➡️ 动作 D：结果解析与收集
            for item_result in result.items_checked:
                process_logs.append(
                    {
                        "check_item": check_item,
                        "is_compliant": item_result.is_compliant,
                        "company_rule": item_result.company_rule_summary,
                        "bid_content": item_result.bid_content_summary,
                    }
                )

                if not item_result.is_compliant:
                    issues_found.append(
                        {
                            "category": f"双向检索核对 ({check_item})",
                            "message": item_result.violation_details or "发现规章冲突",
                            "evidence": item_result.bid_content_summary,
                            "reference_rule": item_result.company_rule_summary,
                        }
                    )

        except Exception as e:
            logger.error(
                f"❌ 双向核对清单项【{check_item}】时发生致命异常: {e}", exc_info=True
            )
            issues_found.append(
                {
                    "category": f"系统错误 ({check_item})",
                    "message": f"核对失败: {str(e)}",
                }
            )

    return {"issues": issues_found, "double_rag_logs": process_logs}
