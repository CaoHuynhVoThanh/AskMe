from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_index: int = Field(description="The 1-based index of the retrieved context chunk.")
    source: str = Field(description="The document source from metadata, if available.")
    excerpt: str = Field(description="A short supporting excerpt.")


class AnswerResponse(BaseModel):
    answer: str = Field(description="The final answer for the user.")
    has_enough_context: bool = Field(description="Whether the retrieved context is sufficient.")
    confidence: float = Field(description="Confidence score from 0 to 1.")
    citations: list[Citation] = Field(default_factory=list, description="Supporting citations.")
    missing_info: list[str] = Field(
        default_factory=list,
        description="Information that is missing when context is insufficient.",
    )
