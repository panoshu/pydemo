from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from demo.agents.compliance import check_compliance_with_rag
from demo.agents.grammar import check_grammar_and_tone
from demo.agents.logic import extract_and_verify_logic
from demo.agents.vision import check_images
from demo.state import ReviewState


def build_async_workflow() -> CompiledStateGraph:
    workflow = StateGraph(ReviewState)

    workflow.add_node("grammar", check_grammar_and_tone)
    workflow.add_node("logic", extract_and_verify_logic)
    workflow.add_node("compliance", check_compliance_with_rag)
    workflow.add_node("vision", check_images)

    # 扇出 (Fan-out)：并发执行
    workflow.add_edge(START, "grammar")
    workflow.add_edge(START, "logic")
    workflow.add_edge(START, "compliance")
    workflow.add_edge(START, "vision")

    # 扇入 (Fan-in)：汇聚结果
    workflow.add_edge("grammar", END)
    workflow.add_edge("logic", END)
    workflow.add_edge("compliance", END)
    workflow.add_edge("vision", END)

    return workflow.compile()
