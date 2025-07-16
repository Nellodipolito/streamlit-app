"""
FastAPI application for medical guidelines QA.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import QuestionRequest, QuestionResponse
from .agent import OrchestratorQAAgent
from fastapi.responses import StreamingResponse

# Create FastAPI app
app = FastAPI(
    title="Medical Guidelines QA API",
    description="API for answering medical questions using guidelines",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize Orchestrator agent
qa_agent = OrchestratorQAAgent()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Medical Guidelines QA API",
        "version": "1.0.0",
        "description": "API for answering medical questions using guidelines",
        "endpoints": {
            "/question": "POST - Ask a medical question",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/question", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """Answer a medical question using the guidelines.
    
    Args:
        request: QuestionRequest containing the question and optional parameters
        
    Returns:
        QuestionResponse containing the answer and related information
    """
    try:
        # Process the question
        result = await qa_agent.answer_question(
            question=request.question,
            top_k=request.top_k
        )
        
        # Convert to response model
        response = QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            chunks=result["chunks"],
            sources=result["sources"],
            search_query=result["search_query"]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 

@app.post("/question/stream")
async def stream_answer_question(request: QuestionRequest):
    """Stream the answer to a medical question using the guidelines."""
    try:
        # Process the question
        result = await qa_agent.answer_question(
            question=request.question,
            top_k=request.top_k
        )
        # result["answer_generator"] is an async generator
        async def answer_stream():
            async for chunk in result["answer_generator"]:
                yield chunk
        return StreamingResponse(answer_stream(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 