"""
FastAPI application for medical guidelines QA.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import QuestionRequest, QuestionResponse, DiagnosisRequest, DiagnosisResponse, FollowUpQuestionRequest, FollowUpQuestionResponse
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
        "description": "API for answering medical questions using guidelines and diagnostic assistance",
        "endpoints": {
            "/question": "POST - Ask a medical question",
            "/question/stream": "POST - Stream answer to a medical question",
            "/diagnosis": "POST - Analyze patient data for potential diagnoses",
            "/diagnosis/stream": "POST - Stream diagnostic analysis",
            "/diagnosis/followup": "POST - Answer follow-up questions about a case",
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

@app.post("/diagnosis", response_model=DiagnosisResponse)
async def analyze_patient(request: DiagnosisRequest):
    """Analyze patient data to provide potential diagnoses.
    
    Args:
        request: DiagnosisRequest containing patient data and analysis parameters
        
    Returns:
        DiagnosisResponse containing diagnostic analysis and evidence
    """
    try:
        # Convert patient data to dict
        patient_data = {
            "patient_summary": request.patient_data.patient_summary,
            "doctor_conversation": request.patient_data.doctor_conversation,
            "age": request.patient_data.age,
            "gender": request.patient_data.gender,
            "chief_complaint": request.patient_data.chief_complaint
        }
        
        # Process the diagnostic request
        result = await qa_agent.diagnostic_agent.diagnose_patient(
            patient_data=patient_data,
            top_k=request.top_k,
            selected_question=request.selected_question
        )
        
        # Collect the complete diagnosis text from the generator
        complete_diagnosis = ""
        async for chunk in result["diagnosis_generator"]:
            complete_diagnosis += chunk
        
        # Parse the completed diagnosis
        parsed_result = qa_agent.diagnostic_agent.parse_completed_diagnosis(
            complete_diagnosis, result["evidence"]
        )
        
        # Convert to response model
        response = DiagnosisResponse(
            patient_summary=result["patient_summary"],
            key_findings=result["key_findings"],
            potential_diagnoses=[
                {
                    "condition": diag["condition"],
                    "reasoning": diag["reasoning"],
                    "supporting_evidence": []  # Could be enhanced
                }
                for diag in parsed_result["parsed_diagnoses"]
            ],
            search_queries=result["search_queries"],
            sources=result["sources"],
            recommendations=complete_diagnosis
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/diagnosis/stream")
async def stream_diagnostic_analysis(request: DiagnosisRequest):
    """Stream the diagnostic analysis response."""
    try:
        # Convert patient data to dict
        patient_data = {
            "patient_summary": request.patient_data.patient_summary,
            "doctor_conversation": request.patient_data.doctor_conversation,
            "age": request.patient_data.age,
            "gender": request.patient_data.gender,
            "chief_complaint": request.patient_data.chief_complaint
        }
        
        # Process the diagnostic request
        result = await qa_agent.diagnostic_agent.diagnose_patient(
            patient_data=patient_data,
            top_k=request.top_k,
            selected_question=request.selected_question
        )
        
        # Stream the diagnosis generation
        async def diagnosis_stream():
            async for chunk in result["diagnosis_generator"]:
                yield chunk
                    
        return StreamingResponse(diagnosis_stream(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/diagnosis/followup", response_model=FollowUpQuestionResponse)
async def answer_followup_question(request: FollowUpQuestionRequest):
    """Answer a follow-up question about a diagnostic case."""
    try:
        result = await qa_agent.diagnostic_agent.answer_followup_question(
            question=request.question,
            case_context=request.case_context,
            top_k=request.top_k
        )
        
        response = FollowUpQuestionResponse(
            question=result["question"],
            answer=result["answer"],
            used_existing_data=result["used_existing_data"],
            sources=result["sources"],
            search_query=result["search_query"]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/diagnosis/templates")
async def get_diagnosis_templates():
    """Get predefined diagnostic questions and templates."""
    return {
        "predefined_questions": [
            "What are the top 5 potential diagnoses for this case?",
            "What are the differential diagnoses to consider?",
            "What risk factors should be evaluated?",
            "What additional tests or evaluations are recommended?",
            "What is the most likely diagnosis based on the presentation?",
            "What red flags or concerning symptoms should be monitored?",
            "What treatment options should be considered?",
            "What is the prognosis for the most likely diagnoses?"
        ],
        "patient_data_template": {
            "patient_summary": "Brief summary of patient's condition, symptoms, and relevant history",
            "doctor_conversation": "Key points from doctor-patient conversation or interview",
            "age": "Patient age (optional)",
            "gender": "Patient gender (optional)",
            "chief_complaint": "Primary reason for visit or main symptom"
        }
    } 