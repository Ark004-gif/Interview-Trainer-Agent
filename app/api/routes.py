"""
FastAPI Application — IBM AICTE Interview Trainer Agent
"""
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agents.interview_agent import interview_agent
from app.api.schemas import (
    CreateSessionRequest,
    EvaluateAnswerRequest,
    EvaluationResponse,
    GenerateQuestionsRequest,
    HealthResponse,
    ModelAnswerRequest,
    ModelAnswerResponse,
    PreparationStrategyRequest,
    QuestionsResponse,
    SessionResponse,
    StrategyResponse,
)
from app.config import settings
from app.rag.rag_pipeline import rag_pipeline

logger = logging.getLogger(__name__)

# ── In-memory session store (keyed by session_id) ────────────────────────────
_sessions: dict[str, dict] = {}

# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="IBM AICTE — Interview Trainer Agent",
        description=(
            "AI-powered interview preparation using IBM Granite (watsonx.ai) "
            "and Retrieval-Augmented Generation (RAG). "
            "Generates tailored questions, model answers, and improvement strategies."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Startup / Shutdown ────────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup():
        # Ensure data directories exist
        Path(settings.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)

        # Initialize RAG pipeline (seeds knowledge base if empty)
        logger.info("Initializing RAG pipeline...")
        rag_pipeline.initialize()
        logger.info("RAG pipeline ready.")

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/api/health", response_model=HealthResponse, tags=["System"])
    async def health():
        kb_count = 0
        try:
            if rag_pipeline._collection:
                kb_count = rag_pipeline._collection.count()
        except Exception:
            pass

        return HealthResponse(
            status="healthy",
            knowledge_base_documents=kb_count,
            ibm_granite_model=settings.IBM_GRANITE_MODEL_ID,
            vector_store=settings.VECTOR_STORE_TYPE,
            version="1.0.0",
        )

    # ── Sessions ──────────────────────────────────────────────────────────────

    @app.post("/api/sessions", response_model=SessionResponse, tags=["Sessions"], status_code=201)
    async def create_session(req: CreateSessionRequest):
        """
        Create a new interview session from a candidate profile.
        Optionally provide resume text for skill extraction.
        """
        session = interview_agent.create_session(
            name=req.name,
            role=req.role,
            experience_level=req.experience_level,
            resume_text=req.resume_text,
        )
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        _sessions[session_id] = session

        return SessionResponse(
            session_id=session_id,
            name=session["name"],
            role=session["role"],
            experience_level=session["experience_level"],
            tech_skills=session["tech_skills"],
            soft_skills=session["soft_skills"],
            resume_summary=session["resume_summary"],
            message=f"Session created for {req.name}. Ready to generate questions.",
        )

    @app.post(
        "/api/sessions/upload",
        response_model=SessionResponse,
        tags=["Sessions"],
        status_code=201,
    )
    async def create_session_with_resume(
        name: str = Form(...),
        role: str = Form(...),
        experience_level: str = Form(...),
        resume: UploadFile = File(...),
    ):
        """
        Create a session by uploading a resume file (PDF, DOCX, or TXT).
        """
        # Validate file type
        allowed = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"}
        if resume.content_type not in allowed and not resume.filename.endswith((".pdf", ".docx", ".txt")):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF, DOCX, and TXT files are supported.",
            )

        # Size check
        contents = await resume.read()
        max_bytes = settings.MAX_RESUME_SIZE_MB * 1024 * 1024
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {settings.MAX_RESUME_SIZE_MB} MB.",
            )

        session = interview_agent.create_session(
            name=name,
            role=role,
            experience_level=experience_level,
            resume_bytes=contents,
            resume_filename=resume.filename,
        )
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        _sessions[session_id] = session

        return SessionResponse(
            session_id=session_id,
            name=session["name"],
            role=session["role"],
            experience_level=session["experience_level"],
            tech_skills=session["tech_skills"],
            soft_skills=session["soft_skills"],
            resume_summary=session["resume_summary"],
            message=f"Resume parsed. Session created for {name}.",
        )

    @app.get("/api/sessions/{session_id}", tags=["Sessions"])
    async def get_session(session_id: str):
        session = _get_session(session_id)
        return {
            "session_id": session_id,
            "name": session["name"],
            "role": session["role"],
            "experience_level": session["experience_level"],
            "tech_skills": session["tech_skills"],
            "soft_skills": session["soft_skills"],
            "questions_generated": len(session.get("questions_generated", [])),
            "answers_evaluated": len(session.get("answers_evaluated", [])),
        }

    # ── Questions ─────────────────────────────────────────────────────────────

    @app.post("/api/questions/generate", response_model=QuestionsResponse, tags=["Interview"])
    async def generate_questions(req: GenerateQuestionsRequest):
        """
        Generate a tailored interview question set using RAG + IBM Granite.
        """
        session = _get_session(req.session_id)
        result = interview_agent.generate_questions(
            session=session,
            num_questions=req.num_questions,
            include_behavioral=req.include_behavioral,
        )
        return QuestionsResponse(session_id=req.session_id, **result)

    # ── Evaluation ────────────────────────────────────────────────────────────

    @app.post("/api/answers/evaluate", response_model=EvaluationResponse, tags=["Interview"])
    async def evaluate_answer(req: EvaluateAnswerRequest):
        """
        Evaluate a candidate's answer with AI-generated feedback and scores.
        """
        session = _get_session(req.session_id)
        result = interview_agent.evaluate_answer(
            question=req.question,
            candidate_answer=req.candidate_answer,
            session=session,
        )
        return EvaluationResponse(session_id=req.session_id, **result)

    # ── Model Answers ─────────────────────────────────────────────────────────

    @app.post("/api/answers/model", response_model=ModelAnswerResponse, tags=["Interview"])
    async def get_model_answer(req: ModelAnswerRequest):
        """
        Get the ideal model answer for an interview question.
        """
        session = _get_session(req.session_id)
        result = interview_agent.get_model_answer(
            question=req.question,
            session=session,
        )
        return ModelAnswerResponse(session_id=req.session_id, **result)

    # ── Preparation Strategy ──────────────────────────────────────────────────

    @app.post("/api/strategy", response_model=StrategyResponse, tags=["Interview"])
    async def get_preparation_strategy(req: PreparationStrategyRequest):
        """
        Generate a personalized 30-day interview preparation roadmap.
        """
        session = _get_session(req.session_id)
        result = interview_agent.generate_preparation_strategy(session=session)
        return StrategyResponse(session_id=req.session_id, **result)

    # ── Utility endpoints ─────────────────────────────────────────────────────

    @app.get("/api/roles", tags=["Utility"])
    async def list_roles():
        return {"roles": settings.SUPPORTED_ROLES}

    @app.get("/api/experience-levels", tags=["Utility"])
    async def list_experience_levels():
        return {"levels": settings.SUPPORTED_EXPERIENCE_LEVELS}

    @app.get("/api/knowledge-base/stats", tags=["Utility"])
    async def kb_stats():
        count = rag_pipeline._collection.count() if rag_pipeline._collection else 0
        return {"total_documents": count, "vector_store": settings.VECTOR_STORE_TYPE}

    # ── Serve Frontend (SPA) ──────────────────────────────────────────────────
    static_dir = Path(__file__).parent.parent / "frontend"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir / "static")), name="static")

        @app.get("/", include_in_schema=False)
        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_frontend(full_path: str = ""):
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(str(index))
            return {"message": "IBM AICTE Interview Trainer Agent API", "docs": "/api/docs"}

    return app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Please create a session first.",
        )
    return session


app = create_app()
