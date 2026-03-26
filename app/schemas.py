from typing import List, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    question: str = Field(min_length=1, description="Original user question")
    answer: str = Field(min_length=1, description="LLM answer payload text")


class BrandAnalysis(BaseModel):
    name: str
    mentions_count: int
    scopes: List[str]
    domain: Optional[str] = None


class AnalysisResponse(BaseModel):
    brands: List[BrandAnalysis]

