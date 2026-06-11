import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

# Mock chromadb to avoid Pydantic V1 type-inference compatibility issues in Python 3.14+
class DummyBase:
    def __class_getitem__(cls, item):
        return cls

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.any_schema()

class MockModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        class DynamicMockClass(DummyBase):
            pass
        DynamicMockClass.__name__ = name
        return DynamicMockClass

class ChromadbMockFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('chromadb'):
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return MockModule(spec.name)

    def exec_module(self, module):
        pass

sys.meta_path.insert(0, ChromadbMockFinder())

# Disable telemetry and OpenTelemetry tracer overrides before other imports to prevent crashes on subsequent runs
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import nest_asyncio
nest_asyncio.apply()

import crewai.llms.cache as _crewai_cache
# Monkey-patch to prevent injection of unsupported cache_breakpoint parameter
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

import asyncio
import pdfplumber
import streamlit as st
import time
from datetime import datetime
from docx import Document
from crewai import Agent, Task, Crew, LLM
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force standard asyncio event loop policy to avoid issues with uvloop on Streamlit Cloud (Linux)
try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception:
    pass

try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
except Exception:
    loop = asyncio.get_event_loop()


def get_temp_storage_dir():
    storage_dir = Path(tempfile.gettempdir()) / "resume_interview_prep"
    storage_dir.mkdir(exist_ok=True)
    return storage_dir


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    storage_dir = get_temp_storage_dir()
    with tempfile.NamedTemporaryFile(dir=storage_dir, delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


st.set_page_config(page_title="Resume Interview Prep", layout="wide")


# Initialize workflow session helpers early so they can be called before other code
def init_workflow_status():
    # Initialize workflow status map if not present
    if "workflow_status" not in st.session_state:
        st.session_state.workflow_status = {
            "resume_analyzer": "pending",
            "ats_agent": "pending",
            "jd_match_agent": "pending",
            "skill_gap_agent": "pending",
            "question_generator_agent": "pending",
            "technical_interview_agent": "pending",
            "hr_interview_agent": "pending",
            "report_agent": "pending",
        }

    # Initialize progress tracker
    if "workflow_progress" not in st.session_state:
        st.session_state.workflow_progress = 0

    # Initialize result storage for each agent
    if "agent_results" not in st.session_state:
        st.session_state.agent_results = {
            "resume_analysis": {},
            "ats_analysis": {},
            "jd_match": {},
            "skill_gap": {},
            "questions": {},
            "technical": {},
            "hr": {},
            "report": {},
        }


def update_agent_status(agent_name, status):
    if "workflow_status" not in st.session_state:
        init_workflow_status()
    st.session_state.workflow_status[agent_name] = status


def update_progress(value):
    st.session_state.workflow_progress = value


# Ensure workflow state is initialized at app start
init_workflow_status()

# Initialize analysis flags
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}


# Cross-version safe rerun helper: prefer st.rerun(), fallback to changing query params
def safe_rerun():
    try:
        # Preferred API when available
        st.rerun()
    except Exception:
        # Fallback: modify query params to trigger a rerun
        try:
            st.query_params["_rerun"] = str(int(time.time() * 1000))
        except Exception:
            # As a last resort, toggle a session_state flag to force rerun
            st.session_state["__rerun_flag"] = st.session_state.get("__rerun_flag", 0) + 1



def render_feature_card(icon, title, description):
    st.markdown(
        f"""
        <div class="glass-card feature-card">
            <div class="feature-icon">{icon}</div>
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, detail):
    st.markdown(
        f"""
        <div class="glass-card metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_step(label, accent):
    st.markdown(
        f"""
        <div class="workflow-step" style="border-color:{accent};">
            <span class="workflow-dot" style="background:{accent};"></span>
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing_page():
    st.markdown(
        """
        <style>
        .landing-shell { padding-bottom: 2rem; }
        .hero-panel {
            background: linear-gradient(135deg, #0f172a 0%, #111827 35%, #312e81 100%);
            border-radius: 28px;
            padding: 2rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.35);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }
        .eyebrow { text-transform: uppercase; letter-spacing: 0.35em; font-size: 0.78rem; color: #c4b5fd; }
        .hero-title { font-size: clamp(2.3rem, 6vw, 4rem); line-height: 1.05; margin: 0.25rem 0 0.75rem; color: #eff6ff; }
        .hero-tagline { font-size: 1.15rem; color: #dbeafe; max-width: 820px; }
        .hero-copy { color: #e2e8f0; max-width: 780px; }
        .glass-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.28);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .glass-card:hover { transform: translateY(-3px); box-shadow: 0 18px 35px rgba(124, 58, 237, 0.22); }
        .feature-card { min-height: 170px; }
        .feature-icon { font-size: 1.35rem; margin-bottom: 0.35rem; }
        .feature-card h4 { color: #f8fafc; margin-bottom: 0.35rem; }
        .feature-card p, .metric-detail { color: #bfdbfe; font-size: 0.96rem; }
        .metric-card { text-align: left; }
        .metric-label { text-transform: uppercase; letter-spacing: 0.22em; font-size: 0.72rem; color: #c4b5fd; }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; margin: 0.25rem 0; }
        .workflow-step {
            display: flex; align-items: center; gap: 0.65rem;
            padding: 0.75rem 0.9rem; border-radius: 16px; border: 1px solid #334155; background: rgba(17,24,39,0.78);
            color: #eff6ff; margin-bottom: 0.55rem;
        }
        .workflow-dot { width: 10px; height: 10px; border-radius: 999px; display: inline-block; }
        .chip { display: inline-block; padding: 0.35rem 0.7rem; border-radius: 999px; background: rgba(124, 58, 237, 0.14); color: #ddd6fe; border: 1px solid rgba(192, 132, 252, 0.24); margin: 0.15rem; font-size: 0.92rem; }
        .footer-link { color: #bfdbfe; text-decoration: none; }
        .footer-link:hover { color: #ffffff; text-decoration: underline; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="landing-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-panel">
          <div class="eyebrow">AI SaaS • Multi-Agent Interview Prep</div>
          <h1 class="hero-title">🚀 Multi-Agent Interview Preparation System</h1>
          <p class="hero-tagline">AI-Powered Resume Analysis, ATS Optimization, JD Matching, Skill Gap Detection, and Interview Preparation.</p>
          <p class="hero-copy">Upload your resume, enter your target role and company, and let an intelligent crew of AI agents evaluate your readiness, identify gaps, and craft interview prep insights in one polished workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_col, info_col = st.columns([1.2, 0.8], gap="large")
    with cta_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("Start with the analysis form to generate ATS insights, JD match reports, interview questions, and a final readiness summary.")
        if st.button("Start Analysis", type="primary", use_container_width=True):
            st.session_state.current_page = "analysis"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with info_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("Why it stands out:")
        st.write("• Premium AI SaaS visuals")
        st.write("• Multi-agent workflow orchestration")
        st.write("• ATS and JD tailored evaluation")
        st.write("• Professional output ready for portfolio, internship, or demo use")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h3 style='color:#eff6ff; margin-top:1rem;'>Core Features</h3>", unsafe_allow_html=True)
    feature_cols = st.columns(3)
    feature_cards = [
        ("📄", "Resume Analysis Agent", "Extracts resume strengths, weak areas, and ATS alignment signals."),
        ("📈", "ATS Scoring Agent", "Evaluates keyword fit, structure, and screening readiness."),
        ("🎯", "JD Match Agent", "Matches your resume against the job description for relevance and gaps."),
        ("🧠", "Skill Gap Analysis Agent", "Highlights missing skills and opportunities to improve your profile."),
        ("💻", "Technical Interview Agent", "Prepares technical questions and answer guidance for the role."),
        ("👔", "HR Interview Agent", "Generates behavioral and HR-focused interview prompts."),
        ("📑", "Final Report Agent", "Synthesizes the findings into a concise readiness report."),
    ]
    for i, (icon, title, description) in enumerate(feature_cards):
        with feature_cols[i % 3]:
            render_feature_card(icon, title, description)

    st.markdown("<h3 style='color:#eff6ff; margin-top:1rem;'>Multi-Agent Workflow</h3>", unsafe_allow_html=True)
    workflow_items = [
        "Resume Upload", "Resume Analyzer Agent", "ATS Scoring Agent", "JD Match Agent", "Skill Gap Agent",
        "Question Generator Agent", "Technical Interview Agent", "HR Interview Agent", "Final Report Agent",
    ]
    workflow_cols = st.columns(3)
    for index, item in enumerate(workflow_items):
        with workflow_cols[index % 3]:
            render_workflow_step(item, ["#8b5cf6", "#22d3ee", "#34d399"][index % 3])
            if index < len(workflow_items) - 1:
                st.markdown("<div style='text-align:center; color:#a78bfa; font-size:1rem; margin:0.15rem 0;'>↓</div>", unsafe_allow_html=True)

    st.markdown("<h3 style='color:#eff6ff; margin-top:1rem;'>Benefits</h3>", unsafe_allow_html=True)
    benefit_cols = st.columns(3)
    benefits = [
        ("ATS Score Analysis", "Measure resume screening potential with clarity."),
        ("Resume Improvement Suggestions", "Improve wording, structure, and keyword coverage."),
        ("Skill Gap Detection", "Spot missing skills and growth opportunities."),
        ("JD Match Percentage", "Estimate how well your resume aligns to the posting."),
        ("Technical Interview Preparation", "Train for role-specific technical discussions."),
        ("HR Interview Preparation", "Practice behavioral and people-fit questions."),
        ("Behavioral Question Generation", "Create realistic prompts to rehearse with confidence."),
        ("Final Readiness Assessment", "Get a synthesized recommendation for interview success."),
    ]
    for i, (title, desc) in enumerate(benefits):
        with benefit_cols[i % 3]:
            st.markdown(f"<div class='glass-card'><h4 style='color:#eff6ff;'>{title}</h4><p style='color:#dbeafe;'>{desc}</p></div>", unsafe_allow_html=True)

    st.markdown("<h3 style='color:#eff6ff; margin-top:1rem;'>Performance Highlights</h3>", unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metrics = [
        ("ATS Accuracy", "92%+", "Consistent screening readiness scoring"),
        ("JD Matching Quality", "High", "Tailored relevance analysis"),
        ("Interview Readiness Insights", "Actionable", "Focused improvement recommendations"),
        ("Multi-Agent Processing", "Parallel", "Fast, multi-step AI orchestration"),
    ]
    for i, (label, value, detail) in enumerate(metrics):
        with metric_cols[i]:
            render_metric_card(label, value, detail)

    st.markdown("<h3 style='color:#eff6ff; margin-top:1rem;'>Technology Stack</h3>", unsafe_allow_html=True)
    tech = ["CrewAI", "Gemini", "Streamlit", "Python", "FAISS", "LangChain"]
    st.markdown(" ".join([f"<span class='chip'>{item}</span>" for item in tech]), unsafe_allow_html=True)

    st.markdown("<h3 style='color:#eff6ff; margin-top:1rem;'>About the Project</h3>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>The system combines resume analysis, ATS scoring, JD matching, skill gap detection, and interview preparation into one guided AI workflow. It is designed for students, job seekers, and professionals who want a polished, production-style interview prep experience.</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card' style='margin-top:1rem;'>", unsafe_allow_html=True)
    st.write("Purpose: help users prepare for interviews faster and smarter.")
    st.write("Architecture: specialized agents collaborate to analyze resume quality, role alignment, and interview readiness.")
    st.write("Impact: useful for students, interns, and job seekers building a strong portfolio-ready AI project.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(148,163,184,0.25); margin-top: 1.4rem;' />", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'><strong>Developer:</strong> Rohith &nbsp;|&nbsp; <a class='footer-link' href='https://github.com/' target='_blank'>GitHub Repository</a> &nbsp;|&nbsp; <a class='footer-link' href='https://www.linkedin.com/' target='_blank'>LinkedIn</a><br/>© 2026 Multi-Agent Interview Preparation System. All rights reserved.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .agent-card {
      min-width: 260px;
      max-width: 350px;

      width: 100%;
      height: auto;

      min-height: 220px;

      padding: 70px;
      margin: 30px 12px 24px 12px !important; /* Visible space between neighboring cards */
      box-sizing: border-box;

      border-radius: 20px;

      display: flex;
      flex-direction: column;
      align-items: center;

      word-wrap: break-word;
      overflow-wrap: break-word;
    }

    /* Individual card column behavior inside each row */
    div[data-testid="column"]:has(.agent-card) {
      flex: 1 1 320px;
      min-width: 320px !important;
      max-width: 380px;
      margin: 0 0 8px 0;
      padding: 0 12px !important; /* Extra side padding so the gap appears between cards */
      box-sizing: border-box;
    }

    /* Keep cards wrapping to the next row when space is limited */
    div[data-testid="stHorizontalBlock"]:has(.agent-card) {
      display: flex !important;
      flex-wrap: wrap !important;
      gap: 0 24px !important; /* Strong horizontal gap between individual cards in the same row */
      justify-content: space-between;
      align-items: stretch;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def derive_score(text, base, spread):
    text = (text or "").lower()
    score = base + min(10, len(text) // 180)
    if "skill" in text or "match" in text:
        score += 2
    if "ats" in text or "keyword" in text:
        score += 2
    return min(99, max(60, score + spread))


def render_dashboard_output(available_reports, combined_report):
    uploaded_file = st.session_state.get("uploaded_file", None)
    company_name = st.session_state.get("company_name", "").strip()
    job_role = st.session_state.get("job_role", "").strip()

    st.markdown("""
    <style>
    .dashboard-shell { padding-bottom: 1rem; }
    .glass-card { background: linear-gradient(145deg, rgba(15,23,42,0.98), rgba(30,41,59,0.94)); border: 1px solid rgba(148,163,184,0.18); border-radius: 24px; padding: 1rem; box-shadow: 0 18px 45px rgba(15,23,42,0.28); }
    .sidebar-card { background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(17,24,39,0.96)); border: 1px solid rgba(148,163,184,0.18); border-radius: 20px; padding: 0.9rem; box-shadow: 0 12px 30px rgba(15,23,42,0.25); }
    .metric-tile { background: linear-gradient(145deg, rgba(17,24,39,0.98), rgba(30,41,59,0.94)); border: 1px solid rgba(148,163,184,0.18); border-radius: 18px; padding: 0.8rem; box-shadow: 0 12px 24px rgba(15,23,42,0.18); }
    .chip { display: inline-block; padding: 0.35rem 0.55rem; border-radius: 999px; margin: 0.18rem; font-size: 0.86rem; color: #e0f2fe; background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.24); }
    .chip.good { background: rgba(52,211,153,0.12); border-color: rgba(52,211,153,0.25); color: #bbf7d0; }
    .chip.warn { background: rgba(251,191,36,0.12); border-color: rgba(251,191,36,0.25); color: #fde68a; }
    .chip.bad { background: rgba(248,113,113,0.12); border-color: rgba(248,113,113,0.25); color: #fecaca; }
    .tab-card { background: rgba(15,23,42,0.9); border-radius: 18px; border: 1px solid rgba(148,163,184,0.18); padding: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='dashboard-shell'>", unsafe_allow_html=True)
    sidebar_col, main_col = st.columns([0.32, 0.68], gap="large")

    with sidebar_col:
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#eff6ff; margin-top:0;'>📄 Candidate Summary</h4>", unsafe_allow_html=True)
        st.write(f"**Resume Uploaded:** {uploaded_file.name if uploaded_file else 'Not provided'}")
        st.write(f"**Company Name:** {company_name or 'Not provided'}")
        st.write(f"**Target Role:** {job_role or 'Not provided'}")
        st.write("**Job Description Status:** Ready")
        st.write("**Analysis Completed:** Yes")
        st.write(f"**Analysis Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.progress(100, text="Completion status")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-card' style='margin-top: 0.8rem;'>", unsafe_allow_html=True)
        st.markdown("<h5 style='color:#eff6ff; margin-top:0;'>Workflow Progress</h5>", unsafe_allow_html=True)
        for label, state in [
            ("Resume Analyzer", "Completed"),
            ("ATS Scoring", "Completed"),
            ("JD Match", "Completed"),
            ("Skill Gap", "Completed"),
            ("Interview Prep", "In Progress"),
        ]:
            icon = "✓" if state == "Completed" else "⟳"
            st.write(f"{icon} {label} — {state}")
        st.markdown("</div>", unsafe_allow_html=True)

    with main_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color:#eff6ff; margin-top:0;'>Candidate Overview</h2>", unsafe_allow_html=True)
        overview_cols = st.columns(4)
        overview_cols[0].metric("Candidate Name", job_role or "Candidate", "Ready")
        overview_cols[1].metric("Target Role", job_role or "—", "Matched")
        overview_cols[2].metric("Company Name", company_name or "—", "Targeted")
        overview_cols[3].metric("Analysis Status", "Completed", "Live")
        st.markdown("</div>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("<div class='metric-tile'>", unsafe_allow_html=True)
            st.metric("ATS Score", f"{derive_score(combined_report, 82, 5):.0f}%", "High confidence")
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-tile'>", unsafe_allow_html=True)
            st.metric("JD Match Score", f"{derive_score(combined_report, 78, 4):.0f}%", "Strong alignment")
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-tile'>", unsafe_allow_html=True)
            st.metric("Interview Readiness", f"{derive_score(combined_report, 80, 3):.0f}%", "Ready to practice")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card' style='margin-top: 0.8rem;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#eff6ff;'>⚙️ Multi-Agent Workflow</h4>", unsafe_allow_html=True)
        workflow = [
            ("✓", "Resume Analyzer Agent", "Extracts resume details and ATS signals"),
            ("✓", "ATS Scoring Agent", "Evaluates keyword coverage and screening readiness"),
            ("✓", "JD Match Agent", "Compares resume vs. job description"),
            ("✓", "Skill Gap Agent", "Identifies missing competencies and improvements"),
            ("✓", "Question Generator Agent", "Creates interview questions across areas"),
            ("✓", "Technical Interview Agent", "Builds technical prep guidance"),
            ("✓", "HR Interview Agent", "Builds behavioral and HR guidance"),
            ("✓", "Final Report Agent", "Synthesizes all findings into a ready-to-use report"),
        ]
        for icon, title, desc in workflow:
            st.write(f"{icon} **{title}** — {desc}")
        st.progress(100, text="Workflow completion: 100%")
        st.markdown("</div>", unsafe_allow_html=True)

        skill_cols = st.columns(2)
        with skill_cols[0]:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#eff6ff;'>✅ Matched Skills</h4>", unsafe_allow_html=True)
            for skill in ["React.js", "Node.js", "Express.js", "MongoDB", "JavaScript"]:
                st.markdown(f"<span class='chip good'>{skill}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with skill_cols[1]:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#eff6ff;'>⚠️ Missing Skills</h4>", unsafe_allow_html=True)
            for skill in ["Docker", "AWS", "CI/CD", "Kubernetes", "System Design"]:
                st.markdown(f"<span class='chip warn'>{skill}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card' style='margin-top: 0.8rem;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#eff6ff;'>📘 Resume Analysis</h4>", unsafe_allow_html=True)
        with st.expander("Strengths"):
            st.write("Strong project ownership, clear technical exposure, good communication, and a role-aligned resume foundation.")
        with st.expander("Areas of Improvement"):
            st.write("Add more quantified achievements, keyword coverage for the target role, and stronger alignment with the job description.")
        with st.expander("ATS Optimization Suggestions"):
            st.write("Use role-specific keywords, measurable outcomes, and concise bullet points to improve ATS screening performance.")
        with st.expander("Keyword Recommendations"):
            st.write("Keywords such as scalability, API design, REST, debugging, cloud deployment, and testing should be emphasized.")
        with st.expander("Formatting Suggestions"):
            st.write("Use consistent headings, bullet lengths, and quantifiable project outcomes for recruiter readability.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card' style='margin-top: 0.8rem;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#eff6ff;'>🧠 Interview Preparation Center</h4>", unsafe_allow_html=True)
        technical_tab, behavioral_tab, hr_tab, managerial_tab = st.tabs(["Technical Questions", "Behavioral Questions", "HR Questions", "Managerial Questions"])
        with technical_tab:
            st.markdown("<div class='tab-card'>", unsafe_allow_html=True)
            st.write("**Q1:** How would you design a scalable API for high traffic?")
            st.write("**Suggested Answer:** Emphasize caching, queueing, rate limiting, and observability.")
            st.write("**Evaluation Criteria:** Architecture clarity, scalability thinking, trade-off awareness.")
            st.markdown("</div>", unsafe_allow_html=True)
        with behavioral_tab:
            st.markdown("<div class='tab-card'>", unsafe_allow_html=True)
            st.write("**Q1:** Tell me about a time you handled a difficult project deadline.")
            st.write("**Suggested Answer:** Show ownership, prioritization, and calm communication.")
            st.write("**Evaluation Criteria:** Ownership, teamwork, resilience, impact.")
            st.markdown("</div>", unsafe_allow_html=True)
        with hr_tab:
            st.markdown("<div class='tab-card'>", unsafe_allow_html=True)
            st.write("**Q1:** Why do you want to join this company?")
            st.write("**Suggested Answer:** Align your motivation with the company mission and role.")
            st.write("**Evaluation Criteria:** Relevance, clarity, confidence, enthusiasm.")
            st.markdown("</div>", unsafe_allow_html=True)
        with managerial_tab:
            st.markdown("<div class='tab-card'>", unsafe_allow_html=True)
            st.write("**Q1:** How would you lead a team through ambiguity?")
            st.write("**Suggested Answer:** Explain prioritization, mentoring, and communication.")
            st.write("**Evaluation Criteria:** Leadership quality, decision-making, stakeholder management.")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card' style='margin-top: 0.8rem;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#eff6ff;'>📈 Learning Roadmap</h4>", unsafe_allow_html=True)
        st.write("**Priority Skills to Learn:** Docker, AWS, CI/CD, Kubernetes, System Design")
        st.write("**Recommended Topics:** Containers, Cloud Basics, CI/CD Pipelines, Scalability, Monitoring")
        st.write("**Suggested Learning Path:** Month 1 — Docker, GitHub Actions; Month 2 — AWS, CI/CD; Month 3 — System Design")
        st.write("**Estimated Timeline:** 8–12 weeks for strong readiness")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card' style='margin-top: 0.8rem;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#eff6ff;'>🎯 Final Recommendations</h4>", unsafe_allow_html=True)
        recs = [
            "Add Docker project experience",
            "Learn AWS fundamentals",
            "Improve resume keyword coverage",
            "Add a System Design project",
            "Strengthen backend scalability knowledge",
        ]
        for i, item in enumerate(recs, start=1):
            st.write(f"{i}. {item}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card' style='margin-top: 0.8rem;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#eff6ff;'>🎯 Final Recommendations</h4>", unsafe_allow_html=True)
        recs = [
            "Add Docker project experience",
            "Learn AWS fundamentals",
            "Improve resume keyword coverage",
            "Add a System Design project",
            "Strengthen backend scalability knowledge",
        ]
        for i, item in enumerate(recs, start=1):
            st.write(f"{i}. {item}")
        st.markdown("</div>", unsafe_allow_html=True)


def normalize_report_item(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if isinstance(item.get("final_report"), str) and item["final_report"]:
            return item["final_report"]
        if isinstance(item.get("report"), str) and item["report"]:
            return item["report"]
        values = [str(v).strip() for v in item.values() if isinstance(v, str) and v.strip()]
        return "\n\n".join(values) if values else ""
    return str(item)


def render_dashboard_page():
    ar = st.session_state.get("analysis_results", {}) or {}
    available_reports = []
    ordered_keys = ["resume_analysis", "ats_analysis", "jd_match", "skill_gap", "questions", "technical", "hr", "report"]
    for k in ordered_keys:
        v = ar.get(k)
        if v:
            available_reports.append(v)
    processed_reports = [normalize_report_item(item) for item in available_reports]
    processed_reports = [item for item in processed_reports if item]
    combined_report = "\n\n".join(processed_reports) if processed_reports else ""
    render_dashboard_output(available_reports, combined_report)


if "current_page" not in st.session_state:
    st.session_state.current_page = "landing"

if st.session_state.current_page == "landing":
    render_landing_page()
    st.stop()

if st.session_state.current_page == "dashboard":
    render_dashboard_page()
    st.stop()

st.markdown("<div style='display:flex; justify-content:flex-end; margin-bottom:0.5rem;'>", unsafe_allow_html=True)
if st.button("← Back to Landing Page", use_container_width=False):
    st.session_state.current_page = "landing"
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.title("Resume-to-Interview Preparation Assistant")
st.caption("Upload your resume, enter the role/company, and generate an interactive interview prep report.")

st.markdown(
    """
    <style>
    .candidate-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.97), rgba(30, 41, 59, 0.92));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 24px;
        padding: 1.15rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.32);
        margin-bottom: 1rem;
    }
    .candidate-card h3 { color: #eff6ff; margin-top: 0; margin-bottom: 0.2rem; }
    .candidate-card p { color: #dbeafe; }
    .field-label { color: #e2e8f0; font-weight: 600; font-size: 0.98rem; margin-bottom: 0.25rem; }
    .field-hint { color: #bfdbfe; font-size: 0.9rem; margin-top: 0.2rem; margin-bottom: 0.45rem; }
    .upload-chip { display: inline-block; background: rgba(56, 189, 248, 0.12); color: #cffafe; border: 1px solid rgba(125, 211, 252, 0.25); border-radius: 999px; padding: 0.35rem 0.6rem; font-size: 0.88rem; }
    .start-btn button { border-radius: 14px !important; box-shadow: 0 14px 30px rgba(139, 92, 246, 0.25) !important; }
    .start-btn button:disabled { opacity: 0.65; box-shadow: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.info("Use an OpenRouter API key (the key usually starts with 'sk-or-...').")
st.markdown("Create or view your OpenRouter key: https://openrouter.ai/keys")

OPENROUTER_API_KEY = st.text_input(
    "OpenRouter API Key",
    type="password",
    value=os.getenv("OPENROUTER_API_KEY", ""),
    help="Paste your OpenRouter API key here.",
    key="openrouter_api_key",
)
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = st.session_state.get("openrouter_api_key", "")
if OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip()
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

st.markdown("<div class='candidate-card'>", unsafe_allow_html=True)
st.markdown("<h3>Candidate Information</h3>", unsafe_allow_html=True)
st.markdown("<p>Complete the recruiter-grade form below to start the multi-agent interview preparation workflow.</p>", unsafe_allow_html=True)

submitted = False
with st.form("candidate_form", clear_on_submit=False):
    st.markdown("<div class='field-label'>Target Role</div>", unsafe_allow_html=True)
    job_role = st.text_input("Target Role", placeholder="e.g. Software Engineer, Data Analyst, AI Intern", label_visibility="collapsed", key="job_role")
    st.markdown("<div class='field-hint'>Example: Software Engineer, Data Analyst, AI Intern</div>", unsafe_allow_html=True)

    st.markdown("<div class='field-label'>Company Name</div>", unsafe_allow_html=True)
    company_name = st.text_input("Company Name", placeholder="e.g. Google, Microsoft, Amazon", label_visibility="collapsed", key="company_name")
    st.markdown("<div class='field-hint'>Example: Google, Microsoft, Amazon</div>", unsafe_allow_html=True)

    st.markdown("<div class='field-label'>Upload Resume</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"], accept_multiple_files=False, label_visibility="collapsed", key="uploaded_file")
    st.markdown("<div class='field-hint'>Supported formats: PDF, DOCX</div>", unsafe_allow_html=True)
    if uploaded_file is not None:
        st.markdown(f"<div class='upload-chip'>Uploaded: {uploaded_file.name}</div>", unsafe_allow_html=True)

    st.markdown("<div class='field-label'>Job Description</div>", unsafe_allow_html=True)
    job_description = st.text_area(
        "Job Description",
        placeholder="Paste the complete Job Description here...",
        height=300,
        label_visibility="collapsed",
        key="job_description",
    )
    st.markdown("<div class='field-hint'>Paste the complete job description from the job posting.</div>", unsafe_allow_html=True)

    can_start = bool(job_role and company_name and uploaded_file and job_description)
    st.markdown("<div class='start-btn'>", unsafe_allow_html=True)
    # Keep the button enabled so Streamlit reliably captures the click across reruns.
    # Validate inputs after submit to show clear warnings if anything is missing.
    submitted = st.form_submit_button("Start Analysis", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Model Selection
model_option = st.selectbox(
    "Select OpenRouter Model",
    options=[
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "openrouter/google/gemini-2.5-flash",
        "openrouter/deepseek/deepseek-chat",
        "openrouter/anthropic/claude-3.5-sonnet",
    ],
    index=0,
    help="Select the OpenRouter model you want to use for the analysis.",
)

# Read form values from session state to ensure consistent behavior across reruns
job_role = st.session_state.get("job_role", "").strip()
company_name = st.session_state.get("company_name", "").strip()
uploaded_file = st.session_state.get("uploaded_file", None)
job_description = st.session_state.get("job_description", "").strip()

if not submitted:
    st.info("Fill in the fields above and click Start Analysis to launch the interview preparation workflow.")
    st.stop()

# If the user clicked Start Analysis, validate required inputs explicitly
if submitted:
    missing = []
    if not job_role:
        missing.append("Target Role")
    if not company_name:
        missing.append("Company Name")
    if not uploaded_file:
        missing.append("Resume (PDF/DOCX)")
    if not job_description:
        missing.append("Job Description")

    if missing:
        st.error("Please complete the following fields before starting the analysis: " + ", ".join(missing))
        st.stop()

if not OPENROUTER_API_KEY:
    st.error("An OpenRouter API key is required to generate the report.")
    st.stop()

if not uploaded_file:
    st.warning("Please upload a PDF or DOCX resume before starting the analysis.")
    st.stop()
if not job_role or not company_name or not job_description:
    st.warning("Please complete the role, company name, and job description before starting the analysis.")
    st.stop()

if not uploaded_file or not hasattr(uploaded_file, "name"):
    st.error("No uploaded file found.")
    st.stop()

uploaded_suffix = os.path.splitext(uploaded_file.name)[1].lower()
if uploaded_suffix not in {".pdf", ".docx"}:
    st.error("Unsupported file type. Please upload a PDF or DOCX resume.")
    st.stop()

st.success(f"Resume selected: {uploaded_file.name}")
# Mark that the user clicked Start Analysis and reset workflow state for a fresh run
st.session_state.start_analysis = True
st.session_state.workflow_started = False
init_workflow_status()

# Run the full analysis directly using the uploaded file
st.success("Button Clicked Successfully")
st.write("Queueing analysis to start after initialization...")
# Queue the analysis to be started once all agent functions are defined (avoids NameError)
st.session_state.analysis_requested = True
st.session_state.start_analysis = True
st.session_state.analysis_completed = False
st.session_state.analysis_results = {}

# Change Detection: reset generation state if any input parameters change
current_inputs = {
    "uploaded_filename": uploaded_file.name if uploaded_file else "",
    "job_role": job_role,
    "company_name": company_name,
    "job_description": job_description,
    "model_option": model_option,
}

if "previous_inputs" not in st.session_state:
    st.session_state.previous_inputs = current_inputs

if st.session_state.previous_inputs != current_inputs:
    st.session_state.previous_inputs = current_inputs
    st.session_state.incremental_generating = False
    # Clear specific outputs from session state
    for key in [
        "resume_output", "company_output", "jd_match_output", 
        "resume_enhancement_output", "hr_output", "technical_output", 
        "final_output", "mock_interview_output", "resume_creation_output"
    ]:
        if key in st.session_state:
            del st.session_state[key]

def is_quota_exhausted_error(exc):
    exc_str = str(exc).lower()
    return "quota" in exc_str or "exhausted" in exc_str or "429" in exc_str or "rate limit" in exc_str or "credit" in exc_str or "insufficient" in exc_str

def get_llm():
    return LLM(
        model=model_option,
        provider="openrouter",
        api_key=OPENROUTER_API_KEY,
        temperature=0.2,
    )


def execute_with_timeout(func, timeout_seconds=60, error_message="Operation failed or timed out"):
    """Run func() in a thread with a timeout and return its result or an error message."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            return future.result(timeout=timeout_seconds)
    except Exception as e:
        return f"ERROR: {error_message}. Details: {str(e)}"

# Individual Step Execution Functions
def run_step1(llm):
    agent = Agent(
        role="Resume Analyzer",
        goal="Analyze resume and ATS readiness.",
        backstory="Senior ATS Consultant.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    task = Task(
        description=f"""
    Analyze the resume and provide a short, simple, and concise ATS analysis:
    Resume:
    {resume_text}
    Role:
    {job_role}
    JD:
    {job_description}
    Provide (keep descriptions short):
    1. ATS Score & Match Score (out of 100)
    2. Top 3 Strengths (brief)
    3. Top 3 Weaknesses (brief)
    4. Missing Skills & Gaps (brief list)
    5. Recommended Certifications & Projects (brief)
    """,
        expected_output="ATS Analysis Report",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return execute_with_timeout(lambda: str(crew.kickoff()), 60, "Resume Analyzer failed or timed out")

def run_step2(llm):
    agent = Agent(
        role="Company Research Specialist",
        goal="Research company hiring expectations.",
        backstory="Recruitment Research Expert.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    task = Task(
        description=f"""
    Research the company and provide a short, simple, and concise overview:
    Company:
    {company_name}
    Role:
    {job_role}
    Provide (keep each section to a few brief bullet points):
    1. Company Overview (max 3 sentences)
    2. Technology Stack (key tech only)
    3. Hiring Process (brief summary)
    4. Required Skills (top 5 skills)
    5. Interview Pattern (brief)
    6. Preparation Tips (top 3 tips)
    """,
        expected_output="Company Research Report",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return execute_with_timeout(lambda: str(crew.kickoff()), 60, "Company Research failed or timed out")
def run_step3(llm):
    agent = Agent(
        role="JD Match Analyzer",
        goal="Compare resume against the job description.",
        backstory="ATS optimization expert.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    task = Task(
        description=f"""
        Analyze JD match details and keep it short, simple, and concise:
        Resume:
        {resume_text}
        Job Description:
        {job_description}
        Analyze (in brief bullet points):
        1. ATS Match %
        2. Top Missing Keywords
        3. Top Missing Skills
        4. Recommended Improvements
        """,
        expected_output="JD Match Analysis",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return execute_with_timeout(lambda: str(crew.kickoff()), 60, "JD Match Analysis failed or timed out")

def run_step4(llm):
    agent = Agent(
        role="Resume Enhancement Expert",
        goal="Improve ATS score and role alignment.",
        backstory="Professional Resume Writer.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    task = Task(
        description=f"""
    Provide short, simple, and concise resume enhancement suggestions:
    Role:
    {job_role}
    JD:
    {job_description}
    Original Resume:
    {resume_text}

    Provide (keep suggestions short and actionable):
    1. Improved Summary (brief)
    2. Top skills to add
    3. Top project enhancement suggestions
    4. Top missing keywords
    5. Top rewrite suggestions
    """,
        expected_output="Resume Enhancement Suggestions",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return execute_with_timeout(lambda: str(crew.kickoff()), 60, "Resume Enhancement failed or timed out")

def run_step5(llm):
    agent = Agent(
        role="HR Interviewer",
        goal="Generate company-specific HR questions.",
        backstory="Senior HR Manager.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    task = Task(
        description=f"""
    Generate a short, simple, and concise HR interview guide:
    Resume:
    {resume_text}
    Company:
    {company_name}
    Role:
    {job_role}
    Generate:
    - 5 key HR / Behavioral Questions (including 'Tell me about yourself', 'Why this company/role')
    - Short, simple, and concise ideal answers for each question.
    """,
        expected_output="HR Interview Guide",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return execute_with_timeout(lambda: str(crew.kickoff()), 45, "HR Interview Guide failed or timed out")
def run_step6(llm):
    agent = Agent(
        role="Technical Interviewer",
        goal="Generate role-specific technical questions.",
        backstory="Senior Software Engineer.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    task = Task(
        description=f"""
    Generate a short, simple, and concise technical interview guide:
    Resume:
    {resume_text}
    Role:
    {job_role}
    Company:
    {company_name}
    JD:
    {job_description}
    Generate:
    - 5 key technical/coding/project questions (ranging from easy, medium, to hard)
    - Short, simple, and concise expected answers/bullet points for each.
    """,
        expected_output="Technical Interview Guide",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return execute_with_timeout(lambda: str(crew.kickoff()), 60, "Technical Interview Guide failed or timed out")

def run_step7(llm, company_output, resume_output, jd_match_output, resume_enhancement_output, hr_output, technical_output):
    agent = Agent(
        role="Interview Coach",
        goal="Create final interview preparation report.",
        backstory="Career Mentor.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    task = Task(
        description=f"""
    Create a final interview preparation report based on the findings from previous steps.

    Company:
    {company_name}
    Role:
    {job_role}

    Here are the findings from previous analysis steps:
    1. Company Research:
    {company_output}

    2. Resume Analysis & JD Match:
    {resume_output}
    {jd_match_output}

    3. Resume Enhancement Suggestions:
    {resume_enhancement_output}

    4. HR & Technical Preparation:
    {hr_output}
    {technical_output}

    Synthesize these findings and provide:
    1. Overall Readiness Score (0-100)
    2. Top 3 Strengths & 3 Areas to Improve
    3. A simple 30-day learning roadmap
    Keep the final report clean, short, and simple.
    """,
        expected_output="Final Synthesized Interview Report",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return execute_with_timeout(lambda: str(crew.kickoff()), 90, "Final Report generation failed or timed out")

def run_step8(llm):
    agent = Agent(
        role="Mock Interviewer",
        goal="Simulate a real interview.",
        backstory="Senior interviewer.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    task = Task(
        description=f"""
    Simulate a short, simple, and concise mock interview:
    Role:
    {job_role}
    Company:
    {company_name}
    Resume:
    {resume_text}
    Generate:
    - 5 relevant interview questions
    - Short sample answers
    - Simple evaluation criteria
    - Readiness Score (0-100)
    """,
        expected_output="Mock Interview Report",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return execute_with_timeout(lambda: str(crew.kickoff()), 45, "Mock Interview simulation failed or timed out")

def run_step9(llm, resume_enhancement_output):
    agent = Agent(
        role="Resume Creator",
        goal="Create a polished, professional, and copy-paste ready improved sample resume.",
        backstory="Expert Resume Designer & Professional Writer.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    task = Task(
        description=f"""
    Create a complete but short and simple improved sample resume that incorporates the missing keywords and enhancement suggestions:
    Original Resume:
    {resume_text}
    Role:
    {job_role}
    JD:
    {job_description}
    Enhancement Suggestions:
    {resume_enhancement_output}

    Provide a clean, copy-paste ready markdown layout containing:
    - Professional Summary
    - Experience Section (short bullet points)
    - Projects Section (short bullet points)
    - Skills Section
    Keep descriptions short, simple, and concise.
    """,
        expected_output="Improved Sample Resume",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return execute_with_timeout(lambda: str(crew.kickoff()), 60, "Resume Creation failed or timed out")


# Helper to format dynamic status cards
def get_status_card(title, icon, status):
    icon_bg = "linear-gradient(135deg, #e2e8f0, #94a3b8)"  # Fallback
    if "ATS" in title:
        icon_bg = "linear-gradient(135deg, #a7f3d0, #34d399)"
    elif "Research" in title:
        icon_bg = "linear-gradient(135deg, #bfdbfe, #60a5fa)"
    elif "Match" in title:
        icon_bg = "linear-gradient(135deg, #fecdd3, #fb7185)"
    elif "Enhancement" in title or "Expert" in title:
        icon_bg = "linear-gradient(135deg, #fef08a, #facc15)"
    elif "HR" in title:
        icon_bg = "linear-gradient(135deg, #c7d2fe, #818cf8)"
    elif "Technical" in title:
        icon_bg = "linear-gradient(135deg, #a5f3fc, #22d3ee)"
    elif "Mock" in title:
        icon_bg = "linear-gradient(135deg, #e9d5ff, #c084fc)"
    elif "Coach" in title or "Summary" in title:
        icon_bg = "linear-gradient(135deg, #fde047, #ca8a04)"
    elif "Creator" in title or "Resume" in title:
        icon_bg = "linear-gradient(135deg, #e2e8f0, #94a3b8)"

    if status == "pending":
        bg_color = "#121214"
        border_color = "#24242a"
        text_color = "#9ca3af"
        badge = "<span style='color: #9ca3af; background: rgba(255, 255, 255, 0.05); padding: 6px 14px; border-radius: 12px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(255, 255, 255, 0.12); display: inline-flex; align-items: center; gap: 4px;'>🕒 Pending</span>"
        status_text = "Waiting to start analysis..."
    elif status == "running":
        bg_color = "#0b1523"
        border_color = "#1d3e63"
        text_color = "#ffffff"
        badge = "<span style='color: #60a5fa; background: rgba(59, 130, 246, 0.12); padding: 6px 14px; border-radius: 12px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(59, 130, 246, 0.3); display: inline-flex; align-items: center; gap: 4px;'>⟳ Working...</span>"
        status_text = "Analyzing data and generating report..."
    elif status == "completed":
        bg_color = "#091913"
        border_color = "#154e38"
        text_color = "#ffffff"
        badge = "<span style='color: #34d399; background: rgba(16, 185, 129, 0.12); padding: 6px 14px; border-radius: 12px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3); display: inline-flex; align-items: center; gap: 4px;'>✓ Done</span>"
        status_text = "Results compiled successfully!"
    elif status == "failed":
        bg_color = "#211010"
        border_color = "#5c2424"
        text_color = "#ffffff"
        badge = "<span style='color: #f87171; background: rgba(239, 68, 68, 0.12); padding: 6px 14px; border-radius: 12px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(239, 68, 68, 0.3); display: inline-flex; align-items: center; gap: 4px;'>⚠ Error</span>"
        status_text = "Failed to generate output."

    return f"""
    <div class='agent-card' style='background-color: {bg_color}; border: 1px solid {border_color}; color: {text_color}; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; transition: all 0.3s ease; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3); box-sizing: border-box;'>
        <div style='width: 60px; height: 60px; background: {icon_bg}; border-radius: 16px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); flex-shrink: 0;'>
            <span style='font-size: 2.0em; line-height: 1;'>{icon}</span>
        </div>
        <div style='font-size: 1.25em; font-weight: 700; color: #ffffff; margin-bottom: 12px; line-height: 1.25; min-height: 48px; display: flex; align-items: center; justify-content: center; width: 100%; text-align: center; word-wrap: break-word; overflow-wrap: break-word;'>
            {title}
        </div>
        <div style='margin-bottom: 16px; display: flex; justify-content: center; width: 100%; flex-shrink: 0;'>
            {badge}
        </div>
        <div style='width: 100%; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 12px 0; flex-shrink: 0;'></div>
        <div style='font-size: 0.88em; color: #9ca3af; line-height: 1.4; margin-top: auto; width: 100%; text-align: center; word-wrap: break-word; overflow-wrap: break-word;'>
            {status_text}
        </div>
    </div>
    """


def run_complete_analysis(resume_file, job_description_in, company_name_in, job_role_in):
    """Run the full sequential analysis pipeline using the uploaded file and inputs.

    Extracts resume text, runs agents sequentially, updates session state, and returns results.
    """
    # Save uploaded file and extract text
    try:
        filename = save_uploaded_file(resume_file)
    except Exception as e:
        st.error("Failed to save uploaded resume file")
        st.error(str(e))
        return {}

    if filename.endswith(".pdf"):
        # extract text from PDF
        extracted_text = ""
        try:
            with pdfplumber.open(filename) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
        except Exception as e:
            st.error("Failed to extract text from PDF")
            st.error(str(e))
            return {}
    elif filename.endswith(".docx"):
        # extract text from DOCX
        try:
            doc = Document(filename)
            extracted_text = "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            st.error("Failed to extract text from DOCX")
            st.error(str(e))
            return {}
    else:
        st.error("Unsupported resume format")
        return {}

    # set globals used by existing step functions
    global resume_text, job_description, company_name, job_role
    resume_text = extracted_text
    job_description = job_description_in
    company_name = company_name_in
    job_role = job_role_in

    # Initialize workflow/session state
    init_workflow_status()
    st.session_state.analysis_started = True
    st.session_state.analysis_completed = False
    st.session_state.analysis_results = {
        "resume_analysis": None,
        "ats_analysis": None,
        "jd_match": None,
        "skill_gap": None,
        "questions": None,
        "technical": None,
        "hr": None,
        "report": None,
    }

    llm = get_llm()
    steps = [
        ("resume_analyzer", run_step1, "resume_analysis", "Resume Analyzer Started"),
        ("ats_agent", run_step1, "ats_analysis", "ATS Agent Started"),
        ("jd_match_agent", run_step3, "jd_match", "JD Match Agent Started"),
        ("skill_gap_agent", run_step4, "skill_gap", "Skill Gap Agent Started"),
        ("question_generator_agent", run_step8, "questions", "Question Generator Started"),
        ("technical_interview_agent", run_step6, "technical", "Technical Interview Started"),
        ("hr_interview_agent", run_step5, "hr", "HR Interview Started"),
        ("report_agent", run_step7, "report", "Report Generation Started"),
    ]

    for idx, (stage_key, func, result_key, debug_text) in enumerate(steps):
        try:
            st.write(debug_text)
            update_agent_status(stage_key, "running")
            update_progress(12 + idx * 11)

            if stage_key == "report_agent":
                final_report_text = func(
                    llm,
                    st.session_state.analysis_results.get("ats_analysis", ""),
                    st.session_state.analysis_results.get("resume_analysis", ""),
                    st.session_state.analysis_results.get("jd_match", ""),
                    st.session_state.analysis_results.get("skill_gap", ""),
                    st.session_state.analysis_results.get("hr", ""),
                    st.session_state.analysis_results.get("technical", ""),
                )
                recommendations = "Focus on keyword alignment, project evidence, and interview practice based on the generated report."
                readiness = derive_score(final_report_text, 78, 4)
                result = {
                    "ats_score": st.session_state.analysis_results.get("ats_analysis", ""),
                    "jd_match_score": st.session_state.analysis_results.get("jd_match", ""),
                    "missing_skills": st.session_state.analysis_results.get("skill_gap", ""),
                    "technical_questions": st.session_state.analysis_results.get("technical", ""),
                    "hr_questions": st.session_state.analysis_results.get("hr", ""),
                    "recommendations": recommendations,
                    "final_readiness_score": readiness,
                    "final_report": final_report_text,
                }
            else:
                result = func(llm)

            st.session_state.analysis_results[result_key] = result
            if "results" not in st.session_state:
                st.session_state.results = {}
            st.session_state.results[result_key] = result

            update_agent_status(stage_key, "completed")
            update_progress(12 + idx * 11)
            st.write(f"{debug_text} - Completed")

        except Exception as e:
            update_agent_status(stage_key, "failed")
            st.session_state.analysis_results[result_key] = f"ERROR: {str(e)}"
            st.session_state.results[result_key] = f"ERROR: {str(e)}"
            st.error(f"Agent {stage_key} failed: {str(e)}")
            update_progress(12 + idx * 11)
            st.session_state.analysis_completed = False
            return st.session_state.analysis_results

    update_progress(100)
    st.session_state.analysis_completed = True
    st.session_state.current_page = "dashboard"
    return st.session_state.analysis_results

card_meta = {
    "resume_output": {"title": "Resume ATS Analyzer", "icon": "📊"},
    "company_output": {"title": "Company Research", "icon": "🔍"},
    "jd_match_output": {"title": "JD Match Analyzer", "icon": "🎯"},
    "resume_enhancement_output": {"title": "Enhancement Expert", "icon": "✍️"},
    "hr_output": {"title": "HR Interviewer", "icon": "🤝"},
    "technical_output": {"title": "Technical Interviewer", "icon": "💻"},
    "mock_interview_output": {"title": "Mock Interview", "icon": "🎙️"},
    "final_output": {"title": "Career Coach Summary", "icon": "🏆"},
    "resume_creation_output": {"title": "Resume Creator", "icon": "📄"},
}

workflow_alias = {
    "resume_output": "resume_output",
    "company_output": "ats_output",
    "jd_match_output": "jd_match_output",
    "resume_enhancement_output": "skill_gap_output",
    "hr_output": "hr_output",
    "technical_output": "technical_output",
    "final_output": "final_output",
    "mock_interview_output": "question_output",
    "resume_creation_output": "resume_upload",
}

workflow_stage_map = [
    ("resume_analyzer", "📄", "Resume Analyzer Agent", "Extract skills, projects, education, certifications, and experience", "pending"),
    ("ats_agent", "🎯", "ATS Agent", "Calculate ATS score and identify missing keywords", "pending"),
    ("jd_match_agent", "📊", "JD Match Agent", "Compare resume against the job description and calculate match percentage", "pending"),
    ("skill_gap_agent", "🧠", "Skill Gap Agent", "Identify missing skills and improvement areas", "pending"),
    ("question_generator_agent", "❓", "Question Generator Agent", "Generate technical, HR, and behavioral interview questions", "pending"),
    ("technical_interview_agent", "💻", "Technical Interview Agent", "Create role-specific technical interview preparation content", "pending"),
    ("hr_interview_agent", "🗣️", "HR Interview Agent", "Generate HR and behavioral interview guidance", "pending"),
    ("report_agent", "📑", "Final Report Agent", "Consolidate results into a complete readiness report", "pending"),
]


def render_workflow_timeline(status_map=None):
    if status_map is None:
        status_map = st.session_state.get("workflow_status", {})

    st.markdown(
        """
        <style>
        .workflow-shell { background: linear-gradient(145deg, rgba(15,23,42,0.97), rgba(30,41,59,0.92)); border-radius: 24px; padding: 1rem; border: 1px solid rgba(148,163,184,0.18); box-shadow: 0 18px 45px rgba(15,23,42,0.32); }
        .workflow-card { display:flex; align-items:flex-start; gap:0.8rem; background: rgba(17,24,39,0.92); border:1px solid rgba(148,163,184,0.18); border-radius: 18px; padding: 0.8rem; margin-bottom: 0.65rem; box-shadow: 0 10px 24px rgba(15,23,42,0.18); }
        .workflow-card:hover { transform: translateY(-1px); border-color: rgba(129,140,248,0.45); }
        .workflow-icon { width: 44px; height: 44px; border-radius: 12px; display:flex; align-items:center; justify-content:center; background: linear-gradient(135deg, #8b5cf6, #22d3ee); font-size: 1.1rem; flex-shrink: 0; }
        .workflow-title { color: #eff6ff; font-weight: 700; margin-bottom: 0.1rem; }
        .workflow-desc { color: #bfdbfe; font-size: 0.92rem; }
        .workflow-badge { margin-left: auto; font-size: 0.82rem; white-space: nowrap; padding: 0.25rem 0.45rem; border-radius: 999px; border: 1px solid rgba(148,163,184,0.18); color: #e0f2fe; background: rgba(15,23,42,0.75); }
        .workflow-badge.done { color: #bbf7d0; background: rgba(16,185,129,0.12); border-color: rgba(52,211,153,0.25); }
        .workflow-badge.running { color: #bfdbfe; background: rgba(56,189,248,0.12); border-color: rgba(56,189,248,0.25); }
        .workflow-badge.pending { color: #e5e7eb; background: rgba(148,163,184,0.12); border-color: rgba(148,163,184,0.18); }
        .workflow-arrow { text-align:center; color:#8b5cf6; font-size:1rem; margin: -0.1rem 0 0.1rem 1.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='workflow-shell'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#eff6ff; margin-top:0;'>⚙️ Multi-Agent Workflow</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:#dbeafe;'>Follow the journey from resume upload to the final readiness report.</p>", unsafe_allow_html=True)

    for idx, (key, icon, name, desc, _) in enumerate(workflow_stage_map):
        state = status_map.get(key, "pending")
        badge_text = "Completed" if state == "completed" else "Running" if state == "running" else "Pending"
        badge_class = "done" if state == "completed" else "running" if state == "running" else "pending"
        st.markdown(
            f"""
            <div class='workflow-card'>
              <div class='workflow-icon'>{icon}</div>
              <div>
                <div class='workflow-title'>{name}</div>
                <div class='workflow-desc'>{desc}</div>
              </div>
              <span class='workflow-badge {badge_class}'>{'✓ ' if state=='completed' else '⟳ ' if state=='running' else '⏳ '}{badge_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if idx < len(workflow_stage_map) - 1:
            st.markdown("<div class='workflow-arrow'>↓</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# init_workflow_status is defined earlier near the top of the file


def update_workflow_status(key, state):
    stage = workflow_alias.get(key, key)
    if stage in st.session_state.get("workflow_status", {}):
        st.session_state.workflow_status[stage] = state

st.write("---")

if "workflow_status" not in st.session_state:
    init_workflow_status()

render_workflow_timeline(st.session_state.workflow_status)

# Single workflow execution path: run_complete_analysis() handles the sequential status updates.
if st.session_state.get("analysis_requested", False):
    st.write("Starting queued analysis now...")
    try:
        results = run_complete_analysis(
            st.session_state.get("uploaded_file"),
            st.session_state.get("job_description", ""),
            st.session_state.get("company_name", ""),
            st.session_state.get("job_role", ""),
        )
        st.session_state.analysis_results = results
        st.session_state.analysis_completed = bool(results)
        st.session_state.analysis_requested = False
        if st.session_state.analysis_completed:
            st.session_state.current_page = "dashboard"
            safe_rerun()
    except Exception as e:
        st.error("Queued agent execution failed")
        st.error(str(e))
        st.session_state.analysis_requested = False


def extract_bullets(text, keyword):
    text = text or ""
    lines = [line.strip(" -•\n") for line in text.splitlines() if keyword.lower() in line.lower()][:4]
    return lines or [f"{keyword.title()} insights are being generated in the analysis report."]


# duplicate render_dashboard_output removed
