# src/demo/workflow.py
import logging

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from demo.agents.compliance import compliance_double_rag_node
from demo.agents.grammar import process_single_chunk
from demo.state import ReviewState

logger = logging.getLogger(__name__)


# ==========================================
# 1. Map-Reduce 动态路由控制器
# ==========================================
def map_chunks_for_review(state: ReviewState):
    """
    【条件边 / Conditional Edge】：决定分配多少个并发 Worker，或直接结束该分支。
    """
    if not state.get("enable_full_text_check", True):
        logger.info("⏭️ 全量审查策略已关闭，跳过 Map-Reduce 分支。")
        return [END]  # 明确告诉 LangGraph 这条并行分支直接结束

    chunks = state.get("document_chunks", [])
    if not chunks:
        logger.warning("⚠️ 未找到标书切片，跳过 Map-Reduce 分支。")
        return [END]

    logger.info(
        f"🌪️ LangGraph 启动 Map-Reduce，分配 {len(chunks)} 个并行片段审查任务..."
    )

    # 核心魔法：返回 Send 指令列表！
    # LangGraph 会自动根据列表长度，克隆出 N 个 "process_chunk_node" 独立并行运行
    return [
        Send("process_chunk_node", {"chunk_index": idx, "chunk_text": chunk})
        for idx, chunk in enumerate(chunks)
    ]


# ==========================================
# 2. 轻量级分发节点
# ==========================================
def dispatch_node(state: ReviewState):
    """
    【分发节点】：一个极轻量的空节点，负责接收 START 的请求并触发后续的并行分支。
    由于我们在 Phase 1 中定义 State 是 Annotated[..., operator.add]，这里不需要返回任何列表。
    """
    logger.info(f"🚦 [Session: {state.get('session_id')}] 开始编排并发审查工作流...")
    return {}


# ==========================================
# 3. 组装企业级 Workflow
# ==========================================
def build_review_graph():
    """
    构建并编译企业级多智能体审查图网络
    """
    workflow = StateGraph(ReviewState)

    # --- A. 注册所有的 Node (计算实体) ---
    workflow.add_node("dispatch", dispatch_node)
    workflow.add_node("compliance_node", compliance_double_rag_node)
    workflow.add_node("process_chunk_node", process_single_chunk)

    # --- B. 定义控制流 (DAG 拓扑) ---
    # 流程起手式：启动 -> 分发节点
    workflow.add_edge(START, "dispatch")

    # 【并行分支 1】：双向 RAG 审查 (基于 Checklist)
    workflow.add_edge("dispatch", "compliance_node")
    workflow.add_edge("compliance_node", END)

    # 【并行分支 2】：全文并发审查 (Map-Reduce 模式)
    workflow.add_conditional_edges(
        "dispatch",
        map_chunks_for_review,
        # 显式声明此条件边可能流向的目标节点（增强健壮性）
        ["process_chunk_node", END],
    )
    # 并发的 N 个 Chunk 处理完毕后，各自汇聚到 END
    workflow.add_edge("process_chunk_node", END)

    # --- C. 编译图网络 ---
    app = workflow.compile()

    return app
