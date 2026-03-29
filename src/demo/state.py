# src/demo/state.py
import operator
from typing import Annotated, Any, Dict, List, TypedDict


class ReviewState(TypedDict):
    """
    企业级 LangGraph 核心状态机：贯穿整个多智能体标书审查生命周期
    """

    # 1. 原始输入数据
    document_text: str
    images_base64: List[str]

    # 2. 审查策略与配置
    checklist: List[str]  # 待核查的关键合规清单
    enable_full_text_check: bool  # 是否执行全量逐段检查
    enable_double_rag_check: bool  # 是否执行双向 RAG 检查

    # 3. 数据预处理结果 (在前置 Node 生成)
    document_chunks: List[str]  # 将超大全文切片后的列表，供全文审查使用
    session_id: str
    # 4. 核心产出数据 (利用 Annotated 和 operator.add 支持并发追加)
    # 所有 Agent 发现的问题最终都会 Reduce 到这里
    issues: Annotated[List[Dict[str, Any]], operator.add]

    # 特定 Agent 的中间执行日志（可选，用于在前端展示流式进度）
    double_rag_logs: Annotated[List[Dict[str, Any]], operator.add]

    # 结构化提取的数据（如金额、工期等）
    extracted_data: Dict[str, Any]

    # 5. 错误捕获
    error_traces: Annotated[List[str], operator.add]
