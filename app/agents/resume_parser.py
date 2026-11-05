"""
Resume Parser
Extracts text and skills from uploaded PDF / DOCX / TXT resumes.
Optionally uses IBM Watson NLU for entity extraction when credentials present.
"""
import io
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

try:
    from docx import Document as DocxDocument
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False


# Common tech skills keyword list for local extraction
_TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "golang", "rust", "c++", "c#",
    "kotlin", "swift", "react", "angular", "vue", "node.js", "django", "fastapi",
    "flask", "spring", "kubernetes", "docker", "terraform", "ansible", "aws", "azure",
    "ibm cloud", "gcp", "postgresql", "mysql", "mongodb", "redis", "kafka", "spark",
    "hadoop", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "sql",
    "nosql", "graphql", "rest", "grpc", "microservices", "ci/cd", "git", "linux",
    "agile", "scrum", "jira", "machine learning", "deep learning", "nlp", "llm",
    "rag", "langchain", "openai", "watsonx", "power bi", "tableau", "excel",
}

_SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "problem solving", "critical thinking",
    "time management", "adaptability", "collaboration", "presentation", "mentoring",
    "stakeholder management", "project management", "negotiation",
}


class ResumeParser:
    """Parses resume files and extracts raw text + skills."""

    # ── Public API ────────────────────────────────────────────────────────────

    def parse_file(self, file_bytes: bytes, filename: str) -> dict:
        """
        Extract text and skills from a resume file.
        Returns dict with keys: raw_text, tech_skills, soft_skills, word_count.
        """
        ext = Path(filename).suffix.lower()
        raw_text = ""

        if ext == ".pdf":
            raw_text = self._extract_pdf(file_bytes)
        elif ext == ".docx":
            raw_text = self._extract_docx(file_bytes)
        elif ext == ".txt":
            raw_text = file_bytes.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return self._analyze(raw_text)

    def parse_text(self, text: str) -> dict:
        """Analyze plain-text resume/job description input."""
        return self._analyze(text)

    # ── Extractors ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_pdf(file_bytes: bytes) -> str:
        if not _PDF_OK:
            return ""
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        if not _DOCX_OK:
            return ""
        doc = DocxDocument(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs)

    # ── Analysis ──────────────────────────────────────────────────────────────

    @staticmethod
    def _analyze(raw_text: str) -> dict:
        lower = raw_text.lower()
        tech = sorted({s for s in _TECH_SKILLS if s in lower})
        soft = sorted({s for s in _SOFT_SKILLS if s in lower})
        years_experience = ResumeParser._estimate_years(raw_text)
        return {
            "raw_text": raw_text,
            "tech_skills": tech,
            "soft_skills": soft,
            "word_count": len(raw_text.split()),
            "estimated_years_experience": years_experience,
        }

    @staticmethod
    def _estimate_years(text: str) -> Optional[int]:
        """Heuristic: find patterns like '5 years', '5+ years', '5-7 years'."""
        pattern = r"(\d{1,2})\+?\s*(?:–|-|to)?\s*(?:\d{1,2}\s*)?years?\s+(?:of\s+)?(?:experience|exp)"
        matches = re.findall(pattern, text.lower())
        if matches:
            return max(int(m) for m in matches)
        return None


resume_parser = ResumeParser()
