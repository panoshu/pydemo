# src/demo/schemas/review.py
from typing import List, Optional

from pydantic import BaseModel, Field


# ==========================================
# 接口请求与响应模型
# ==========================================
class ReviewRequest(BaseModel):
    document_text: str = Field(..., description="标书全文或核心段落文本")
    images_base64: List[str] = Field(
        default_factory=list, description="附带的资质图片 base64 列表"
    )
    checklist: List[str] = Field(
        default_factory=lambda: [
            "付款比例与周期",
            "项目交付或实施周期",
            "违约金及赔偿条款",
            "核心资质要求（如 ISO 认证、系统集成资质等）",
        ],
        description="双向 RAG 审查的重点关注清单",
    )
    # 策略开关，生产环境中可根据文件大小自动决定
    enable_full_text_check: bool = Field(
        default=True, description="是否开启全量逐段审查 (耗时较长)"
    )
    enable_double_rag_check: bool = Field(
        default=True, description="是否开启基于清单的双向 RAG 审查"
    )


# ==========================================
# Agent 输出的结构化约束模型 (强类型防护)
# ==========================================
class IssueItem(BaseModel):
    category: str = Field(
        ..., description="问题分类，如 '语法拼写', '合规红线', '逻辑矛盾'"
    )
    message: str = Field(..., description="问题详细描述")
    evidence: Optional[str] = Field(None, description="标书原文中的证据片段")
    reference_rule: Optional[str] = Field(
        None, description="违反的公司内部规章条款 (如有)"
    )


class BaseAgentResult(BaseModel):
    has_issues: bool = Field(..., description="是否发现了问题")
    issues: List[IssueItem] = Field(default_factory=list, description="发现的问题列表")


class GrammarAgentResult(BaseAgentResult):
    pass


class DoubleRAGItemResult(BaseModel):
    check_item: str = Field(..., description="当前检查的清单项")
    is_compliant: bool = Field(..., description="该项是否合规")
    bid_content_summary: str = Field(..., description="标书中关于此项的描述摘要")
    company_rule_summary: str = Field(..., description="公司关于此项的规章约束")
    violation_details: Optional[str] = Field(
        None, description="如果不合规，详细说明原因"
    )


class ComplianceAgentResult(BaseAgentResult):
    items_checked: List[DoubleRAGItemResult] = Field(
        default_factory=list, description="各个检查项的具体比对结果"
    )
