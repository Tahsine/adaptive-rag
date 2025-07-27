from typing import Literal
from pydantic import BaseModel, Field

class RouteQuery(BaseModel):
    """Route a user query to a specific data source."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user query, choose to route it to web search or vectorstore."
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Document are relevant to question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )