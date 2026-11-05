"""
Interview Trainer Agent — Core Logic
Orchestrates RAG retrieval + IBM Granite generation to produce:
  - Tailored question sets
  - Model answers
  - Preparation strategies
  - Answer evaluations
"""
import logging
from typing import Optional

from app.agents.resume_parser import resume_parser
from app.knowledge_base.interview_data import (
    BEHAVIORAL_QUESTIONS,
    EXPERIENCE_LEVEL_GUIDE,
    HR_GUIDELINES,
    IMPROVEMENT_TIPS,
    TECHNICAL_QUESTIONS,
)
from app.llm.granite_client import granite_client
from app.rag.rag_pipeline import rag_pipeline

logger = logging.getLogger(__name__)


# ── Prompt Templates ──────────────────────────────────────────────────────────

_QUESTION_GENERATION_PROMPT = """You are an expert interview coach and technical interviewer.

Candidate Profile:
- Name: {name}
- Job Role: {role}
- Experience Level: {experience_level}
- Key Skills: {skills}
- Resume Summary: {resume_summary}

Relevant Interview Knowledge:
{context}

Task: Generate {num_questions} highly tailored interview questions for this candidate.
Include a mix of:
- Technical questions specific to their role and skills
- Behavioral questions (use STAR format)
- Situational / problem-solving questions
- At least 1 system design or architecture question (if senior level)

Format each question as:
Q[N]. [Question]
Category: [technical/behavioral/situational/system_design]
Difficulty: [easy/intermediate/hard]
Why this question: [1-sentence rationale based on the candidate's profile]

Be specific, challenging, and relevant to {role} at the {experience_level} level.
"""

_PREPARATION_STRATEGY_PROMPT = """You are a senior career coach specializing in technical hiring.

Candidate Profile:
- Name: {name}
- Role: {role}
- Experience Level: {experience_level}
- Technical Skills: {tech_skills}
- Soft Skills: {soft_skills}

Interview Knowledge Context:
{context}

Task: Create a comprehensive, personalized interview preparation strategy for {name}.

Cover:
1. **30-Day Preparation Roadmap** (week-by-week plan)
2. **Technical Deep-Dives** (top 5 topics to master for {role})
3. **Behavioral Story Bank** (5 STAR stories to prepare)
4. **Common Pitfalls** to avoid for {experience_level} {role} candidates
5. **Day-Of Checklist** (virtual and in-person tips)
6. **Confidence-Building Techniques**

Make it actionable, specific to {role} and {experience_level}, and prioritized by impact.
"""

_ANSWER_EVALUATION_PROMPT = """You are an experienced interviewer evaluating a candidate's response.

Question: {question}
Candidate's Answer: {answer}
Job Role: {role}
Experience Level: {experience_level}

Relevant Best Practices:
{context}

Evaluate the answer on:
1. **Technical Accuracy** (0-10): Is the content correct and deep enough?
2. **Communication Clarity** (0-10): Is it well-structured and easy to follow?
3. **Completeness** (0-10): Does it fully address the question?
4. **Relevance** (0-10): Is it tailored to the role/level?
5. **Use of Examples** (0-10): Are concrete examples provided?

Then provide:
- **Overall Score**: [X/50]
- **Strengths**: 2-3 specific things done well
- **Areas for Improvement**: 2-3 specific, actionable suggestions
- **Improved Answer**: A rewritten version incorporating all improvements
- **Star Rating**: [1-5 stars] with one-line summary

Be constructive, specific, and encouraging.
"""

_MODEL_ANSWER_PROMPT = """You are an expert technical interviewer.

Question: {question}
Job Role: {role}
Experience Level: {experience_level}

Relevant Context:
{context}

Provide a comprehensive model answer that:
1. Is appropriately detailed for {experience_level} level
2. Includes concrete examples and real-world applications
3. Uses structured frameworks (STAR, problem-solution-result) where appropriate
4. Highlights the key insight an interviewer is looking for
5. Is deliverable in 2-3 minutes verbally

Also provide:
- **Key Points to Hit** (3-5 bullet points)
- **What NOT to Say** (1-2 common mistakes)
- **Follow-up Questions** the interviewer might ask
"""


class InterviewTrainerAgent:
    """
    Core agent that orchestrates the full interview preparation pipeline.
    """

    # ── Session Management ────────────────────────────────────────────────────

    def create_session(
        self,
        name: str,
        role: str,
        experience_level: str,
        resume_text: Optional[str] = None,
        resume_bytes: Optional[bytes] = None,
        resume_filename: Optional[str] = None,
    ) -> dict:
        """
        Create a new interview session for a candidate.
        Returns the parsed profile dict.
        """
        # Parse resume
        parsed = {}
        if resume_bytes and resume_filename:
            parsed = resume_parser.parse_file(resume_bytes, resume_filename)
        elif resume_text:
            parsed = resume_parser.parse_text(resume_text)

        tech_skills = parsed.get("tech_skills", [])
        soft_skills = parsed.get("soft_skills", [])
        raw_text = parsed.get("raw_text", resume_text or "")

        # Auto-detect experience level from resume if not overridden
        if parsed.get("estimated_years_experience") and experience_level == "auto":
            yrs = parsed["estimated_years_experience"]
            experience_level = self._years_to_level(yrs)

        session = {
            "name": name,
            "role": role,
            "experience_level": experience_level,
            "tech_skills": tech_skills,
            "soft_skills": soft_skills,
            "resume_summary": self._summarize_resume(raw_text, role),
            "questions_generated": [],
            "answers_evaluated": [],
        }
        logger.info("Session created for %s | Role: %s | Level: %s", name, role, experience_level)
        return session

    # ── Question Generation ───────────────────────────────────────────────────

    def generate_questions(
        self,
        session: dict,
        num_questions: int = 10,
        include_behavioral: bool = True,
    ) -> dict:
        """
        Generate a tailored question set using RAG + IBM Granite.
        Returns dict with ai_questions and curated knowledge-base questions.
        """
        role = session["role"]
        level = session["experience_level"]

        # RAG retrieval
        query = f"Interview questions for {role} at {level} level with skills: {', '.join(session['tech_skills'][:8])}"
        context = rag_pipeline.retrieve_context_string(query, role=role, n_results=8)

        # IBM Granite generation
        prompt = _QUESTION_GENERATION_PROMPT.format(
            name=session["name"],
            role=role,
            experience_level=level,
            skills=", ".join((session["tech_skills"] + session["soft_skills"])[:10]) or "Not specified",
            resume_summary=session.get("resume_summary", "Not provided"),
            context=context,
            num_questions=num_questions,
        )
        ai_questions_text = granite_client.generate(prompt)

        # Curated questions from knowledge base
        kb_technical = TECHNICAL_QUESTIONS.get(role, [])
        kb_behavioral = BEHAVIORAL_QUESTIONS[:5] if include_behavioral else []

        # Filter KB questions by difficulty alignment
        level_difficulty_map = {
            "fresher": "easy",
            "junior": "easy",
            "mid-level": "intermediate",
            "senior": "intermediate",
            "lead": "hard",
            "principal": "hard",
        }
        target_difficulty = level_difficulty_map.get(level, "intermediate")
        filtered_technical = [
            q for q in kb_technical
            if q.get("difficulty") in (target_difficulty, "intermediate")
        ][:6]

        session["questions_generated"] = filtered_technical

        return {
            "ai_generated_questions": ai_questions_text,
            "curated_technical_questions": filtered_technical,
            "curated_behavioral_questions": kb_behavioral,
            "total_curated": len(filtered_technical) + len(kb_behavioral),
            "context_used": context[:500] + "..." if len(context) > 500 else context,
        }

    # ── Preparation Strategy ──────────────────────────────────────────────────

    def generate_preparation_strategy(self, session: dict) -> dict:
        """
        Generate a personalized 30-day preparation roadmap using RAG + Granite.
        """
        role = session["role"]
        level = session["experience_level"]

        query = f"Interview preparation strategy roadmap for {role} {level}"
        context = rag_pipeline.retrieve_context_string(query, role=role, n_results=8)

        prompt = _PREPARATION_STRATEGY_PROMPT.format(
            name=session["name"],
            role=role,
            experience_level=level,
            tech_skills=", ".join(session["tech_skills"][:10]) or "General",
            soft_skills=", ".join(session["soft_skills"][:5]) or "General",
            context=context,
        )
        strategy_text = granite_client.generate(prompt)

        # Static enrichment from knowledge base
        exp_guide = EXPERIENCE_LEVEL_GUIDE.get(level, {})
        improvement = IMPROVEMENT_TIPS

        return {
            "ai_strategy": strategy_text,
            "experience_expectations": exp_guide.get("expectations", []),
            "focus_areas": exp_guide.get("focus_areas", []),
            "salary_tip": exp_guide.get("salary_negotiation_tip", ""),
            "hr_dos": HR_GUIDELINES["dos"],
            "hr_donts": HR_GUIDELINES["donts"],
            "improvement_tips": improvement,
            "virtual_interview_tips": HR_GUIDELINES["virtual_interview_tips"],
        }

    # ── Answer Evaluation ─────────────────────────────────────────────────────

    def evaluate_answer(
        self,
        question: str,
        candidate_answer: str,
        session: dict,
    ) -> dict:
        """
        Evaluate a candidate's answer against best practices using RAG + Granite.
        Returns structured feedback with scores and improvement suggestions.
        """
        role = session["role"]
        level = session["experience_level"]

        query = f"Best answer for: {question} Role: {role}"
        context = rag_pipeline.retrieve_context_string(query, role=role, n_results=5)

        prompt = _ANSWER_EVALUATION_PROMPT.format(
            question=question,
            answer=candidate_answer,
            role=role,
            experience_level=level,
            context=context,
        )
        evaluation_text = granite_client.generate(prompt)

        session["answers_evaluated"].append(
            {"question": question, "answer": candidate_answer, "feedback": evaluation_text}
        )

        return {
            "question": question,
            "candidate_answer": candidate_answer,
            "evaluation": evaluation_text,
            "role": role,
            "level": level,
        }

    # ── Model Answer ──────────────────────────────────────────────────────────

    def get_model_answer(self, question: str, session: dict) -> dict:
        """
        Generate a model/ideal answer for a given question using RAG + Granite.
        """
        role = session["role"]
        level = session["experience_level"]

        query = f"Model answer for interview question: {question}"
        context = rag_pipeline.retrieve_context_string(query, role=role, n_results=5)

        # Check KB for existing model answer first
        for q in TECHNICAL_QUESTIONS.get(role, []):
            if q["question"].lower() in question.lower() or question.lower() in q["question"].lower():
                return {
                    "question": question,
                    "model_answer": q["model_answer"],
                    "tips": q.get("tips", ""),
                    "source": "knowledge_base",
                }

        prompt = _MODEL_ANSWER_PROMPT.format(
            question=question,
            role=role,
            experience_level=level,
            context=context,
        )
        answer_text = granite_client.generate(prompt)

        return {
            "question": question,
            "model_answer": answer_text,
            "tips": "",
            "source": "ibm_granite",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _summarize_resume(text: str, role: str) -> str:
        """Return a short (200-char) summary of resume text."""
        if not text:
            return f"No resume provided. Preparing for {role} role."
        words = text.split()[:60]
        return " ".join(words) + ("..." if len(text.split()) > 60 else "")

    @staticmethod
    def _years_to_level(years: int) -> str:
        if years <= 0:
            return "fresher"
        if years <= 2:
            return "junior"
        if years <= 5:
            return "mid-level"
        if years <= 9:
            return "senior"
        if years <= 14:
            return "lead"
        return "principal"


# Singleton
interview_agent = InterviewTrainerAgent()
