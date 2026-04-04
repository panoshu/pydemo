# src/demo/agents/chat.py
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from demo.config.factory import get_async_llm
from demo.rag.engine import get_company_rules_pipeline


# 1. 升级状态机：引入 Annotated 和 add_messages
class ChatGraphState(TypedDict):
    # 核心：使用 add_messages 聚合器，新消息会自动 append，保留历史记忆
    messages: Annotated[list, add_messages]
    use_rag: bool
    sources: List[Dict[str, Any]]
    context_str: str


# 2. 检索节点 (Retrieve)
async def retrieve_node(state: ChatGraphState):
    if not state.get("use_rag"):
        return {"sources": [], "context_str": ""}

    # 获取用户的最新问题（messages 列表的最后一个）
    latest_query = state["messages"][-1].content

    pipeline = get_company_rules_pipeline()
    nodes = await pipeline.aretrieve(latest_query)

    sources = [
        {
            "text": n.node.get_content()[:200] + "...",
            "score": round(n.score, 4) if n.score else None,
            "file_name": n.node.metadata.get("file_name", "未知文件"),
        }
        for n in nodes
    ]
    context_str = "\n\n---\n\n".join([n.node.get_content() for n in nodes])

    return {"sources": sources, "context_str": context_str}


# 3. 生成节点 (Generate)
async def generate_node(state: ChatGraphState):
    llm = get_async_llm()
    messages_to_send = state["messages"].copy()

    if state.get("use_rag") and state.get("context_str"):
        # 将 RAG 检索到的知识作为 SystemMessage 插入到对话最前面
        sys_prompt = f"""你是一个严谨的企业规章制度助手。请基于以下检索到的【参考资料】回答问题。
要求：严格依据参考资料回答，绝不编造。如果资料中没有，明确回答“抱歉，未检索到相关规定”。

【参考资料】:
{state["context_str"]}
"""
        messages_to_send.insert(0, SystemMessage(content=sys_prompt))

    # 直接把包含了历史记忆的 messages_to_send 交给大模型
    response = await llm.ainvoke(messages_to_send)

    # 只需要返回大模型的最新回复，add_messages 会自动将其合并到全局状态的 messages 列表中
    return {"messages": [response]}


# 4. 组装具备记忆的 Graph
# 初始化一个内存 Checkpointer（生产环境可以换成 AsyncPostgresSaver 或 Redis）
memory = MemorySaver()


def build_chat_graph():
    workflow = StateGraph(ChatGraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # 🌟 关键：编译图网络时注入 checkpointer
    return workflow.compile(checkpointer=memory)


# 全局单例的 app 实例，保证内存持久化生效
chat_app = build_chat_graph()
