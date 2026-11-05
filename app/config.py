"""
IBM AICTE Interview Trainer Agent
Configuration loader — reads settings from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


class Settings:
    # ── IBM watsonx.ai ──────────────────────────────────────────────────────
    IBM_WATSONX_API_KEY: str = os.getenv("IBM_WATSONX_API_KEY", "")
    IBM_WATSONX_PROJECT_ID: str = os.getenv("IBM_WATSONX_PROJECT_ID", "")
    IBM_WATSONX_URL: str = os.getenv(
        "IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
    )
    IBM_GRANITE_MODEL_ID: str = os.getenv(
        "IBM_GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2"
    )

    # ── IBM Watson Discovery ────────────────────────────────────────────────
    IBM_WATSON_DISCOVERY_API_KEY: str = os.getenv("IBM_WATSON_DISCOVERY_API_KEY", "")
    IBM_WATSON_DISCOVERY_URL: str = os.getenv("IBM_WATSON_DISCOVERY_URL", "")
    IBM_WATSON_DISCOVERY_PROJECT_ID: str = os.getenv(
        "IBM_WATSON_DISCOVERY_PROJECT_ID", ""
    )

    # ── IBM Watson NLU ──────────────────────────────────────────────────────
    IBM_WATSON_NLU_API_KEY: str = os.getenv("IBM_WATSON_NLU_API_KEY", "")
    IBM_WATSON_NLU_URL: str = os.getenv("IBM_WATSON_NLU_URL", "")

    # ── App ─────────────────────────────────────────────────────────────────
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    APP_DEBUG: bool = os.getenv("APP_DEBUG", "False").lower() == "true"
    APP_SECRET_KEY: str = os.getenv("APP_SECRET_KEY", "dev-secret-key")

    # ── Vector Store ────────────────────────────────────────────────────────
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chromadb")
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "interview_knowledge_base"
    )

    # ── LLM Generation Params ───────────────────────────────────────────────
    LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "1024"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.95"))
    LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", "50"))
    LLM_REPETITION_PENALTY: float = float(
        os.getenv("LLM_REPETITION_PENALTY", "1.1")
    )

    # ── Agent ───────────────────────────────────────────────────────────────
    MAX_QUESTIONS_PER_SESSION: int = int(
        os.getenv("MAX_QUESTIONS_PER_SESSION", "15")
    )
    DEFAULT_DIFFICULTY: str = os.getenv("DEFAULT_DIFFICULTY", "intermediate")
    SUPPORTED_ROLES: list = os.getenv(
        "SUPPORTED_ROLES",
        "Software Engineer,Data Scientist,Product Manager,DevOps Engineer,"
        "ML Engineer,Full Stack Developer,Backend Developer,Frontend Developer,"
        "Cloud Architect,Business Analyst",
    ).split(",")
    SUPPORTED_EXPERIENCE_LEVELS: list = os.getenv(
        "SUPPORTED_EXPERIENCE_LEVELS",
        "fresher,junior,mid-level,senior,lead,principal",
    ).split(",")

    # ── File Upload ─────────────────────────────────────────────────────────
    MAX_RESUME_SIZE_MB: int = int(os.getenv("MAX_RESUME_SIZE_MB", "5"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")

    # ── Logging ─────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/app.log")


settings = Settings()
