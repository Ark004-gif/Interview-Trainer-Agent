"""
IBM AICTE -- Interview Trainer Agent
Streamlit Application
Run: streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -- Page config (must be first Streamlit call) --------------------------------
st.set_page_config(
    page_title="IBM AICTE -- Interview Trainer Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Lazy-load heavy modules after page config ---------------------------------
from app.agents.interview_agent import interview_agent
from app.agents.resume_parser import resume_parser
from app.config import settings
from app.knowledge_base.interview_data import (
    BEHAVIORAL_QUESTIONS,
    EXPERIENCE_LEVEL_GUIDE,
    HR_GUIDELINES,
    IMPROVEMENT_TIPS,
    TECHNICAL_QUESTIONS,
)
from app.rag.rag_pipeline import rag_pipeline

# -- Custom CSS ----------------------------------------------------------------
st.markdown("""
<style>
/* IBM Carbon Design tokens */
:root {
    --ibm-blue: #0f62fe;
    --ibm-blue-dark: #0043ce;
    --ibm-gray-90: #262626;
    --ibm-gray-10: #f4f4f4;
    --ibm-green: #198038;
    --ibm-red: #da1e28;
    --ibm-purple: #6929c4;
    --ibm-teal: #007d79;
}
/* Top header bar */
.ibm-header {
    background: #161616;
    color: white;
    padding: 0.6rem 1.5rem;
    border-radius: 4px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.ibm-logo-badge {
    background: #0f62fe;
    color: white;
    padding: 2px 10px;
    font-weight: 900;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    border-radius: 2px;
}
.ibm-header-title { font-size: 1rem; font-weight: 600; letter-spacing: 0.02em; }
.ibm-header-sub   { font-size: 0.75rem; color: #8d8d8d; margin-left: auto; }

/* Status pill */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 700;
}
.status-green  { background: #defbe6; color: #198038; }
.status-yellow { background: #fdf6dd; color: #b28600; }
.status-red    { background: #fff1f1; color: #da1e28; }
.status-blue   { background: #d0e2ff; color: #0043ce; }

/* Section dividers */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6f6f6f;
    margin-bottom: 0.3rem;
}
/* Answer box */
.answer-box {
    background: #f0f4ff;
    border-left: 4px solid #0f62fe;
    padding: 1rem 1.2rem;
    border-radius: 0 4px 4px 0;
    font-size: 0.9rem;
    line-height: 1.8;
    white-space: pre-wrap;
}
/* Metric strip */
.metric-strip {
    background: #f4f4f4;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #0f62fe; }
.metric-lbl { font-size: 0.72rem; color: #6f6f6f; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# -- RAG init (cached so it only runs once per session) ------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def init_rag():
    rag_pipeline.initialize()
    return rag_pipeline

rag = init_rag()

# -- Session state defaults ----------------------------------------------------
for key, default in {
    "session": None,
    "questions_result": None,
    "eval_history": [],
    "strategy_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -- Header --------------------------------------------------------------------
st.markdown("""
<div class="ibm-header">
  <span class="ibm-logo-badge">IBM</span>
  <span class="ibm-header-title">AICTE &mdash; Interview Trainer Agent</span>
  <span class="ibm-header-sub">Powered by IBM Granite (watsonx.ai) &amp; RAG &bull; ChromaDB</span>
</div>
""", unsafe_allow_html=True)

# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "",
        [" Create Session", " Question Bank",
         " Practice & Evaluate", " Prep Strategy",
         " Model Answers", " HR Guide"],
        label_visibility="collapsed",
    )

    st.divider()

    # Active session badge
    if st.session_state.session:
        s = st.session_state.session
        st.success(f"**Active Session**\n\n**{s['name']}**\n\n{s['role']} . {s['experience_level']}")
        if s.get("tech_skills"):
            st.caption("Skills detected: " + ", ".join(s["tech_skills"][:5]))
    else:
        st.info("No active session.\nGo to **Create Session** to start.")

    st.divider()
    st.caption(f"Model: `{settings.IBM_GRANITE_MODEL_ID}`")
    kb_count = rag._collection.count() if rag._collection else 0
    st.caption(f"KB docs: {kb_count} | Store: ChromaDB")


#
# PAGE: Create Session
#
if page == " Create Session":
    st.title("Create Interview Session")
    st.markdown("Tell us about yourself to receive a personalised experience.")

    tab_text, tab_upload = st.tabs([" Text Input", " Resume Upload"])

    with tab_text:
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name *", placeholder="e.g. Aryan Sharma")
        with col2:
            role = st.selectbox("Target Role *", [""] + settings.SUPPORTED_ROLES)

        level = st.selectbox(
            "Experience Level *",
            ["fresher", "junior", "mid-level", "senior", "lead", "principal"],
            index=2,
            format_func=lambda x: {
                "fresher": "Fresher (0-1 yr)", "junior": "Junior (1-3 yrs)",
                "mid-level": "Mid-Level (3-6 yrs)", "senior": "Senior (6-10 yrs)",
                "lead": "Lead (10-15 yrs)", "principal": "Principal (15+ yrs)",
            }[x],
        )
        resume_text = st.text_area(
            "Resume / Job Description (optional)",
            placeholder="Paste your resume text, LinkedIn summary, or job description here for personalised questions...",
            height=150,
        )
        if st.button(" Create Session", type="primary", use_container_width=True):
            if not name or not role:
                st.error("Please fill in Name and Role.")
            else:
                with st.spinner("Creating session and parsing profile..."):
                    session = interview_agent.create_session(
                        name=name, role=role, experience_level=level,
                        resume_text=resume_text or None,
                    )
                    st.session_state.session = session
                st.success(f"Session created for **{name}**!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Role", role)
                c2.metric("Level", level)
                c3.metric("Skills Found", len(session.get("tech_skills", [])))
                if session.get("tech_skills"):
                    st.markdown("**Detected Tech Skills:** " + "  ".join(
                        [f'`{s}`' for s in session["tech_skills"]]
                    ))

    with tab_upload:
        col1, col2 = st.columns(2)
        with col1:
            u_name = st.text_input("Full Name *", placeholder="e.g. Aryan Sharma", key="uname")
        with col2:
            u_role = st.selectbox("Target Role *", [""] + settings.SUPPORTED_ROLES, key="urole")
        u_level = st.selectbox(
            "Experience Level *",
            ["fresher", "junior", "mid-level", "senior", "lead", "principal"],
            index=2, key="ulevel",
            format_func=lambda x: {
                "fresher": "Fresher (0-1 yr)", "junior": "Junior (1-3 yrs)",
                "mid-level": "Mid-Level (3-6 yrs)", "senior": "Senior (6-10 yrs)",
                "lead": "Lead (10-15 yrs)", "principal": "Principal (15+ yrs)",
            }[x],
        )
        uploaded = st.file_uploader("Upload Resume (PDF, DOCX, TXT -- max 5 MB)", type=["pdf", "docx", "txt"])
        if st.button(" Upload & Create Session", type="primary", use_container_width=True):
            if not u_name or not u_role:
                st.error("Please fill in Name and Role.")
            else:
                with st.spinner("Parsing resume and creating session..."):
                    kwargs = dict(name=u_name, role=u_role, experience_level=u_level)
                    if uploaded:
                        kwargs["resume_bytes"] = uploaded.read()
                        kwargs["resume_filename"] = uploaded.name
                    session = interview_agent.create_session(**kwargs)
                    st.session_state.session = session
                st.success(f"Session created for **{u_name}**!")
                if session.get("tech_skills"):
                    st.markdown("**Detected Tech Skills:** " + "  ".join(
                        [f'`{s}`' for s in session["tech_skills"]]
                    ))


# 
# PAGE: Question Bank
# 
elif page == " Question Bank":
    st.title("Question Bank")
    st.markdown("AI-generated + curated role-specific interview questions.")

    if not st.session_state.session:
        st.warning(" No active session. Go to **Create Session** first.")
        st.stop()

    s = st.session_state.session
    st.info(f"Session: **{s['name']}** | {s['role']} | {s['experience_level']}")

    col1, col2 = st.columns(2)
    num_q = col1.selectbox("Number of questions", [5, 10, 15], index=1)
    inc_beh = col2.selectbox("Include behavioral", ["Yes", "No -- Technical Only"]) == "Yes"

    if st.button(" Generate with IBM Granite", type="primary", use_container_width=True):
        with st.spinner("Retrieving from knowledge base and generating with IBM Granite..."):
            result = interview_agent.generate_questions(
                session=s, num_questions=num_q, include_behavioral=inc_beh
            )
            st.session_state.questions_result = result

    if st.session_state.questions_result:
        r = st.session_state.questions_result

        # AI-generated (only show when Granite returned content)
        if r["ai_generated_questions"].strip():
            st.subheader(" AI-Generated Questions (IBM Granite)")
            st.markdown('<div class="section-label">IBM watsonx.ai . Granite</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{r["ai_generated_questions"]}</div>', unsafe_allow_html=True)
            st.divider()

        # Curated Technical
        st.subheader(f" Curated Technical Questions ({len(r['curated_technical_questions'])})")
        for i, q in enumerate(r["curated_technical_questions"], 1):
            with st.expander(f"Q{i}. {q['question']}", expanded=False):
                col_a, col_b = st.columns(2)
                col_a.markdown(f"**Category:** `{q.get('category','technical')}`")
                col_b.markdown(f"**Difficulty:** `{q.get('difficulty','intermediate')}`")
                st.markdown("**Model Answer:**")
                st.info(q.get("model_answer", ""))
                if q.get("tips"):
                    st.markdown(f" **Tip:** {q['tips']}")
                if st.button(" Practice this question", key=f"prac_tech_{i}"):
                    st.session_state["prefill_question"] = q["question"]
                    st.info("Question copied! Switch to **Practice & Evaluate** page.")

        # Behavioral
        if r["curated_behavioral_questions"]:
            st.divider()
            st.subheader(f" Behavioral Questions -- STAR Framework ({len(r['curated_behavioral_questions'])})")
            for i, q in enumerate(r["curated_behavioral_questions"], 1):
                with st.expander(f"B{i}. {q['question']}", expanded=False):
                    st.markdown(f"**Framework:** `{q.get('framework','STAR')}`")
                    st.markdown("**Guidance:**")
                    st.info(q.get("model_answer", ""))
                    if q.get("tips"):
                        st.markdown(f" **Tip:** {q['tips']}")
                    if st.button(" Practice this question", key=f"prac_beh_{i}"):
                        st.session_state["prefill_question"] = q["question"]
                        st.info("Question copied! Switch to **Practice & Evaluate** page.")


# 
# PAGE: Practice & Evaluate
# 
elif page == " Practice & Evaluate":
    st.title("Practice & Evaluate")
    st.markdown("Submit your answer to any question and get instant AI feedback with scores.")

    if not st.session_state.session:
        st.warning(" No active session. Go to **Create Session** first.")
        st.stop()

    s = st.session_state.session

    # Pre-fill from question bank if available
    default_q = st.session_state.pop("prefill_question", "")

    question = st.text_area(
        "Interview Question",
        value=default_q,
        placeholder="Enter or paste the interview question here...",
        height=80,
    )
    answer = st.text_area(
        "Your Answer",
        placeholder="Type your answer here. Be detailed -- the AI evaluates depth, clarity, examples, and role-fit...",
        height=180,
    )

    if st.button(" Evaluate My Answer", type="primary", use_container_width=True):
        if not question or not answer:
            st.error("Please fill in both the question and your answer.")
        elif len(answer.strip()) < 20:
            st.error("Answer is too short. Please provide a detailed response.")
        else:
            with st.spinner("Evaluating with IBM Granite..."):
                result = interview_agent.evaluate_answer(
                    question=question, candidate_answer=answer, session=s
                )
                st.session_state.eval_history.append(result)

    if st.session_state.eval_history:
        latest = st.session_state.eval_history[-1]
        st.divider()
        st.subheader(" Evaluation Feedback")
        if latest["evaluation"].strip():
            st.markdown('<div class="section-label">IBM watsonx.ai . Granite Analysis</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{latest["evaluation"]}</div>', unsafe_allow_html=True)
        else:
            st.info("IBM Granite is currently unavailable. Your answer has been recorded. Curated questions and model answers are still accessible.")

        if len(st.session_state.eval_history) > 1:
            st.divider()
            with st.expander(f" Evaluation History ({len(st.session_state.eval_history)} answers)"):
                for i, ev in enumerate(reversed(st.session_state.eval_history[:-1]), 1):
                    st.markdown(f"**#{i}** -- {ev['question'][:80]}...")
                    st.caption(ev["evaluation"][:200] + "...")
                    st.divider()


# 
# PAGE: Prep Strategy
# 
elif page == " Prep Strategy":
    st.title("Preparation Strategy")
    st.markdown("Your personalised 30-day interview readiness roadmap.")

    if not st.session_state.session:
        st.warning(" No active session. Go to **Create Session** first.")
        st.stop()

    s = st.session_state.session

    if st.button(" Generate My Preparation Strategy", type="primary", use_container_width=True):
        with st.spinner("Building your personalised strategy with IBM Granite..."):
            st.session_state.strategy_result = interview_agent.generate_preparation_strategy(session=s)

    if st.session_state.strategy_result:
        r = st.session_state.strategy_result

        # AI Roadmap (only show when Granite returned content)
        if r["ai_strategy"].strip():
            st.subheader(" AI Preparation Roadmap (IBM Granite)")
            st.markdown(f'<div class="answer-box">{r["ai_strategy"]}</div>', unsafe_allow_html=True)
            st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Level Expectations")
            for e in r["experience_expectations"]:
                st.markdown(f"- {e}")

            st.subheader(" Focus Areas")
            st.markdown("  ".join([f"`{a}`" for a in r["focus_areas"]]))

        with col2:
            st.subheader(" Interview Do's")
            for d in r["hr_dos"]:
                st.markdown(f"- {d}")

        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            st.subheader(" Interview Don'ts")
            for d in r["hr_donts"]:
                st.markdown(f"- {d}")
        with col4:
            st.subheader(" Salary Negotiation Tip")
            st.info(r["salary_tip"])

        st.divider()
        st.subheader(" Virtual Interview Tips")
        for tip in r["virtual_interview_tips"]:
            st.markdown(f"- {tip}")


# 
# PAGE: Model Answers
# 
elif page == " Model Answers":
    st.title("Model Answers")
    st.markdown("Get the ideal answer for any interview question.")

    if not st.session_state.session:
        st.warning(" No active session. Go to **Create Session** first.")
        st.stop()

    s = st.session_state.session
    question = st.text_area(
        "Interview Question",
        placeholder="Enter the question you want a model answer for...",
        height=90,
    )

    if st.button(" Generate Model Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Retrieving model answer..."):
                result = interview_agent.get_model_answer(question=question, session=s)

            if result["model_answer"].strip():
                source_label = " Source: Knowledge Base" if result["source"] == "knowledge_base" else " Source: IBM Granite"
                st.caption(source_label)
                st.subheader(" Model Answer")
                st.markdown(f'<div class="answer-box">{result["model_answer"]}</div>', unsafe_allow_html=True)
                if result.get("tips"):
                    st.divider()
                    st.markdown(f" **Pro Tip:** {result['tips']}")
            else:
                st.info("IBM Granite is currently unavailable. Try a question that matches our knowledge base for an instant answer.")


# 
# PAGE: HR Guide
# 
elif page == " HR Guide":
    st.title("HR Interview Guide")
    st.markdown("Best practices, dos & don'ts, and salary negotiation tips.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Always Do")
        for d in HR_GUIDELINES["dos"]:
            st.markdown(f"- {d}")
    with col2:
        st.subheader(" Never Do")
        for d in HR_GUIDELINES["donts"]:
            st.markdown(f"- {d}")

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader(" Salary Negotiation")
        for tip in HR_GUIDELINES["salary_negotiation"]:
            st.markdown(f"- {tip}")
    with col4:
        st.subheader(" Virtual Interview Tips")
        for tip in HR_GUIDELINES["virtual_interview_tips"]:
            st.markdown(f"- {tip}")

    st.divider()
    st.subheader(" Improvement Tips")
    cats = list(IMPROVEMENT_TIPS.keys())
    tip_cols = st.columns(len(cats))
    for col, cat in zip(tip_cols, cats):
        col.markdown(f"**{cat.title()}**")
        for tip in IMPROVEMENT_TIPS[cat]:
            col.markdown(f"- {tip}")


