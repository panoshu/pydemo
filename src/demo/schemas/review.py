from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    document_text: str = Field(..., description="待审核的标书文本")
    images_base64: list[str] = Field(default_factory=list, description="图片Base64列表")


class ReviewResponse(BaseModel):
    status: str
    has_issues: bool
    issues: list[str]
    extracted_data: dict
