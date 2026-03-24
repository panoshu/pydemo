import operator
from typing import Annotated, Any, List, TypedDict


# 安全的字典合并 Reducer
def merge_dicts(left: dict, right: dict) -> dict:
    if not left:
        left = {}
    if not right:
        right = {}
    return {**left, **right}


# 统一的 Issue 结构
class ReviewIssue(TypedDict):
    category: str
    message: str


class ReviewState(TypedDict):
    document_text: str

    # 【核心优化 1】丢弃 Base64，改用文件路径，彻底解决 State 内存爆炸
    image_paths: List[str]

    # 【核心优化 2】统一所有 Agent 的输出通道，符合开闭原则
    issues: Annotated[List[ReviewIssue], operator.add]

    # 【核心优化 3】使用自定义 Reducer，防止多个 Agent 提取数据时互相覆盖
    extracted_data: Annotated[dict[str, Any], merge_dicts]
