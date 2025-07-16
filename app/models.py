from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionRequest(BaseModel):
    """Request model for question answering."""
    question: str = Field(..., description="The medical question to answer")
    top_k: Optional[int] = Field(default=10, description="Number of chunks to retrieve", ge=1, le=50)

class Source(BaseModel):
    """Model for source information."""
    title: str
    year: Optional[int]
    journal: Optional[str]
    section: Optional[str]

class Chunk(BaseModel):
    """Model for retrieved chunks."""
    title: str
    year: Optional[int]
    journal: Optional[str]
    section: Optional[str]
    content: str

class QuestionResponse(BaseModel):
    """Response model for question answering."""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The generated answer")
    chunks: List[Chunk] = Field(..., description="The retrieved chunks used to generate the answer")
    sources: List[Source] = Field(..., description="The sources used to generate the answer")
    search_query: str = Field(..., description="The search query used to retrieve chunks") 