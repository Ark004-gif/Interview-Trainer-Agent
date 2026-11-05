"""
Pydantic request/response schemas for the Interview Trainer API.
"""
from typing import Optional
from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, example="Aryan Sharma")
    role: str = Field(..., example="Software Engineer")
    experience_level: str = Field(
        ...,
        example="mid-level",
        description="fresher | junior | mid-level | senior | lead | principal | auto",
    )
    resume_text: Optional[str] = Field(
        None,
        max_length=20000,
        description="Paste resume text directly (alternative to file upload)",
    )


class GenerateQuestionsRequest(BaseModel):
    session_id: str = Field(..., example="sess_abc123")
    num_questions: int = Field(10, ge=3, le=20)
    include_behavioral: bool = True


class EvaluateAnswerRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=10)
    candidate_answer: str = Field(..., min_length=20, max_length=5000)


class ModelAnswerRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=10)


class PreparationStrategyRequest(BaseModel):
    session_id: str


# ── Response Models ───────────────────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str
    name: str
    role: str
    experience_level: str
    tech_skills: list[str]
    soft_skills: list[str]
    resume_summary: str
    message: str


class QuestionsResponse(BaseModel):
    session_id: str
    ai_generated_questions: str
    curated_technical_questions: list[dict]
    curated_behavioral_questions: list[dict]
    total_curated: int
    context_used: str


class EvaluationResponse(BaseModel):
    session_id: str
    question: str
    candidate_answer: str
    evaluation: str
    role: str
    level: str


class ModelAnswerResponse(BaseModel):
    session_id: str
    question: str
    model_answer: str
    tips: str
    source: str


class StrategyResponse(BaseModel):
    session_id: str
    ai_strategy: str
    experience_expectations: list[str]
    focus_areas: list[str]
    salary_tip: str
    hr_dos: list[str]
    hr_donts: list[str]
    improvement_tips: dict
    virtual_interview_tips: list[str]


class HealthResponse(BaseModel):
    status: str
    knowledge_base_documents: int
    ibm_granite_model: str
    vector_store: str
    version: str
