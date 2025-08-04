from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

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
    link: Optional[str] = Field(default=None, description="Link to the source document")

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

class PatientData(BaseModel):
    """Model for patient information."""
    patient_summary: str = Field(..., description="Summary of patient's condition and history")
    doctor_conversation: Optional[str] = Field(default="", description="Conversation between doctor and patient")
    age: Optional[int] = Field(default=None, description="Patient age")
    gender: Optional[str] = Field(default=None, description="Patient gender")
    chief_complaint: Optional[str] = Field(default="", description="Primary reason for visit")

class DiagnosisRequest(BaseModel):
    """Request model for diagnostic analysis."""
    patient_data: PatientData = Field(..., description="Patient information and data")
    selected_question: Optional[str] = Field(default=None, description="Specific diagnostic question to focus on")
    query_type: str = Field(default="diagnosis", description="Type of diagnostic query", 
                           pattern="^(diagnosis|differential|risk_factors|treatment|prognosis)$")
    top_k: Optional[int] = Field(default=10, description="Number of evidence chunks to retrieve", ge=1, le=50)

class FollowUpQuestionRequest(BaseModel):
    """Request model for follow-up diagnostic questions."""
    question: str = Field(..., description="The follow-up question to answer")
    case_context: Dict = Field(..., description="Previous diagnostic analysis context")
    top_k: Optional[int] = Field(default=10, description="Number of evidence chunks to retrieve", ge=1, le=50)

class FollowUpQuestionResponse(BaseModel):
    """Response model for follow-up questions."""
    question: str = Field(..., description="The follow-up question")
    answer: str = Field(..., description="The answer to the follow-up question")
    used_existing_data: bool = Field(..., description="Whether existing data was sufficient")
    sources: List[Source] = Field(..., description="Sources used for the answer")
    search_query: Optional[str] = Field(default=None, description="Search query if new search was needed")

class DiagnosisEvidence(BaseModel):
    """Model for diagnostic evidence from sources."""
    condition: str = Field(..., description="Medical condition or diagnosis")
    evidence: str = Field(..., description="Supporting evidence from medical literature")
    source_title: str = Field(..., description="Title of the source document")
    source_year: Optional[int] = Field(default=None, description="Year of the source")
    confidence: str = Field(..., description="Confidence level of the evidence")

class PotentialDiagnosis(BaseModel):
    """Model for a potential diagnosis."""
    condition: str = Field(..., description="Name of the condition")
    reasoning: str = Field(..., description="Clinical reasoning for this diagnosis")
    supporting_evidence: List[DiagnosisEvidence] = Field(default=[], description="Evidence supporting this diagnosis")

class DiagnosisResponse(BaseModel):
    """Response model for diagnostic analysis."""
    patient_summary: str = Field(..., description="Processed patient summary")
    key_findings: List[str] = Field(..., description="Key clinical findings extracted")
    potential_diagnoses: List[PotentialDiagnosis] = Field(..., description="Ranked potential diagnoses")
    search_queries: List[str] = Field(..., description="Search queries used to find evidence")
    sources: List[Source] = Field(..., description="All sources consulted")
    recommendations: str = Field(..., description="Clinical recommendations and next steps") 

class CitedAnswer(BaseModel):
    """Answer based only on the given sources, with citations."""
    answer: str = Field(..., description="Answer grounded in the sources")
    citations: List[int] = Field(..., description="IDs of sources used") 