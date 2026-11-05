"""
RAG Pipeline — ChromaDB vector store + IBM watsonx.ai Granite embeddings
"""
import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings
from app.knowledge_base.interview_data import (
    BEHAVIORAL_QUESTIONS,
    EXPERIENCE_LEVEL_GUIDE,
    HR_GUIDELINES,
    IMPROVEMENT_TIPS,
    SOFT_SKILLS_CRITERIA,
    TECHNICAL_QUESTIONS,
)

logger = logging.getLogger(__name__)


class InterviewRAGPipeline:
    """
    Manages the ChromaDB vector store for retrieval-augmented generation.
    Uses sentence-transformers (all-MiniLM-L6-v2) for fast local embeddings;
    retrieved context is then passed to IBM Granite for generation.
    """

    def __init__(self):
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    # ── Setup ────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create or load the ChromaDB collection and seed it if empty."""
        db_path = Path(settings.CHROMA_DB_PATH)
        db_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        if self._collection.count() == 0:
            logger.info("Seeding knowledge base with interview data...")
            self._seed_knowledge_base()
            logger.info(
                "Knowledge base seeded with %d documents.", self._collection.count()
            )
        else:
            logger.info(
                "Knowledge base loaded (%d documents).", self._collection.count()
            )

    def _seed_knowledge_base(self) -> None:
        """Insert all role-specific and behavioral knowledge into ChromaDB."""
        documents, metadatas, ids = [], [], []
        doc_id = 0

        # Technical questions per role
        for role, questions in TECHNICAL_QUESTIONS.items():
            for q in questions:
                text = (
                    f"Role: {role}\n"
                    f"Question: {q['question']}\n"
                    f"Model Answer: {q['model_answer']}\n"
                    f"Tips: {q.get('tips', '')}"
                )
                documents.append(text)
                metadatas.append(
                    {
                        "type": "technical_question",
                        "role": role,
                        "category": q.get("category", "technical"),
                        "difficulty": q.get("difficulty", "intermediate"),
                    }
                )
                ids.append(f"tech_{doc_id}")
                doc_id += 1

        # Behavioral questions
        for q in BEHAVIORAL_QUESTIONS:
            text = (
                f"Behavioral Question: {q['question']}\n"
                f"Framework: {q.get('framework', 'STAR')}\n"
                f"Model Answer: {q['model_answer']}\n"
                f"Tips: {q.get('tips', '')}"
            )
            documents.append(text)
            metadatas.append({"type": "behavioral_question", "role": "all", "category": "behavioral"})
            ids.append(f"behav_{doc_id}")
            doc_id += 1

        # Soft skills
        for skill, data in SOFT_SKILLS_CRITERIA.items():
            for aq in data["assessment_questions"]:
                text = (
                    f"Soft Skill: {skill}\n"
                    f"Description: {data['description']}\n"
                    f"Assessment Question: {aq}"
                )
                documents.append(text)
                metadatas.append({"type": "soft_skill", "skill": skill, "role": "all"})
                ids.append(f"soft_{doc_id}")
                doc_id += 1

        # Experience level guides
        for level, data in EXPERIENCE_LEVEL_GUIDE.items():
            text = (
                f"Experience Level: {level}\n"
                f"Expectations: {'; '.join(data['expectations'])}\n"
                f"Focus Areas: {', '.join(data['focus_areas'])}\n"
                f"Salary Tip: {data['salary_negotiation_tip']}"
            )
            documents.append(text)
            metadatas.append({"type": "experience_guide", "level": level, "role": "all"})
            ids.append(f"exp_{doc_id}")
            doc_id += 1

        # HR guidelines
        hr_text = (
            f"HR Interview Do's: {'; '.join(HR_GUIDELINES['dos'])}\n"
            f"HR Interview Don'ts: {'; '.join(HR_GUIDELINES['donts'])}\n"
            f"Salary Negotiation: {'; '.join(HR_GUIDELINES['salary_negotiation'])}\n"
            f"Virtual Interview Tips: {'; '.join(HR_GUIDELINES['virtual_interview_tips'])}"
        )
        documents.append(hr_text)
        metadatas.append({"type": "hr_guidelines", "role": "all"})
        ids.append(f"hr_{doc_id}")
        doc_id += 1

        # Improvement tips
        for category, tips in IMPROVEMENT_TIPS.items():
            text = f"Improvement Tips ({category}): {'; '.join(tips)}"
            documents.append(text)
            metadatas.append({"type": "improvement_tips", "category": category, "role": "all"})
            ids.append(f"imp_{doc_id}")
            doc_id += 1

        # Batch insert
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            self._collection.add(
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size],
            )

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        role: Optional[str] = None,
        n_results: int = 6,
    ) -> list[dict]:
        """
        Retrieve the top-n most relevant documents for a query.
        Optionally filter by role using ChromaDB where clause.
        """
        if self._collection is None:
            raise RuntimeError("RAG pipeline not initialized. Call initialize() first.")

        where_filter = None
        if role:
            where_filter = {
                "$or": [
                    {"role": {"$eq": role}},
                    {"role": {"$eq": "all"}},
                ]
            }

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, self._collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(
                {
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": round(1 - dist, 4),  # cosine similarity
                }
            )
        return retrieved

    def retrieve_context_string(
        self, query: str, role: Optional[str] = None, n_results: int = 6
    ) -> str:
        """Return retrieved documents as a single concatenated context string."""
        docs = self.retrieve(query, role=role, n_results=n_results)
        return "\n\n---\n\n".join(d["content"] for d in docs)


# Singleton instance
rag_pipeline = InterviewRAGPipeline()
