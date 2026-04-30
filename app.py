"""
AI Resume Screening System — v2 (OOP + Sentence Transformers)

Key upgrades over v1:
  ✅ Replaced GloVe (.kv binary) with sentence-transformers/all-MiniLM-L6-v2
     → No large binary files needed; model auto-downloads on first run
     → Cached via @st.cache_resource — loads once per deployment
  ✅ Added __init__.py to all packages → fixes Streamlit Cloud ImportError
  ✅ JD paste → auto-extract required skills
  ✅ Adjustable threshold slider
  ✅ Unmatched skill gap view per candidate
  ✅ One-click Excel download of shortlisted candidates
  ✅ Analytics tab: score distribution, top-N chart, skill coverage, degree pie
  ✅ Structured Candidate dataclass (name, email, phone, degree, exp)
"""

import glob
import io
import os
import tempfile
import zipfile
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── OOP layer imports ────────────────────────────────────────────────────────
from domain.candidate import Candidate
from infrastructure.file_reader import FileReader
from infrastructure.ml_engine import MLEngine
from services.parser_service import ParserService
from services.screener_service import ScreenerService
from services.skill_matcher import SkillMatcher

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Header */
    .app-title { font-size: 2rem; font-weight: 800; color: #1565C0; letter-spacing: -0.5px; }
    .app-sub   { color: #546e7a; font-size: 0.95rem; margin-top: -8px; }

    /* Score badges */
    .badge-high { background:#e8f5e9; color:#2e7d32; border-radius:6px;
                  padding:2px 10px; font-weight:700; font-size:1.05rem; }
    .badge-med  { background:#fff8e1; color:#f57f17; border-radius:6px;
                  padding:2px 10px; font-weight:700; font-size:1.05rem; }
    .badge-low  { background:#ffebee; color:#c62828; border-radius:6px;
                  padding:2px 10px; font-weight:700; font-size:1.05rem; }

    /* Skill pills */
    .pill-match   { display:inline-block; background:#e3f2fd; color:#0d47a1;
                    border-radius:12px; padding:2px 10px; margin:2px;
                    font-size:0.82rem; }
    .pill-missing { display:inline-block; background:#fce4ec; color:#880e4f;
                    border-radius:12px; padding:2px 10px; margin:2px;
                    font-size:0.82rem; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached model loader (runs once per deployment) ────────────────────────────
@st.cache_resource(show_spinner="🤖  Loading AI model — one-time setup, ~30 s …")
def _load_transformer() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Dependency injection ─────────────────────────────────────────────────────
@st.cache_resource
def _build_services():
    engine = MLEngine()
    engine.set_model(_load_transformer())
    fr  = FileReader()
    ps  = ParserService()
    sm  = SkillMatcher(engine)
    sc  = ScreenerService(fr, sm, ps)
    return sc, ps


try:
    screener, parser_service = _build_services()
except Exception as exc:
    st.error(f"❌ Startup error: {exc}")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    threshold = st.slider(
        "Minimum score threshold (%)", min_value=20, max_value=90, value=60, step=5,
        help="Candidates below this score are filtered out in results."
    )
    show_all = st.checkbox(
        "Show all candidates (ignore threshold)", value=False
    )
    top_n = st.slider(
        "Max candidates to display", min_value=5, max_value=50, value=20, step=5
    )

    st.divider()
    st.markdown("**📊 Scoring formula**")
    st.markdown("| Component | Weight |")
    st.markdown("|-----------|--------|")
    st.markdown("| 🎯 Skills | **55 %** |")
    st.markdown("| 💼 Experience | **25 %** |")
    st.markdown("| 🎓 Qualification | **20 %** |")

    st.divider()
    st.markdown("**🔬 AI model**")
    st.code("all-MiniLM-L6-v2\n(Sentence Transformers)", language="text")
    st.markdown("3-layer matching: Exact → Fuzzy → Semantic")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">📄 AI Resume Screening System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Semantic skill matching · Regex contact parsing · '
    'Automated shortlisting</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_screen, tab_analytics = st.tabs(["🔍  Screen Resumes", "📊  Analytics"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCREENING
# ════════════════════════════════════════════════════════════════════════════
with tab_screen:
    left, right = st.columns([1.1, 0.9], gap="large")

    # ── Left column: requirements ─────────────────────────────────────────
    with left:
        st.subheader("📝 Job Requirements")

        use_jd = st.toggle("Paste a Job Description to auto-extract skills")

        if use_jd:
            jd_text = st.text_area(
                "Job Description",
                height=160,
                placeholder="Paste the full JD here. We'll extract required skills automatically.",
            )
            if st.button("🔍 Extract Skills from JD", use_container_width=True):
                if jd_text.strip():
                    extracted = parser_service.extract_skills_from_jd(jd_text)
                    if extracted:
                        st.session_state["auto_skills"] = ", ".join(extracted)
                        st.success(f"✅ Extracted **{len(extracted)}** skills from JD!")
                    else:
                        st.warning("⚠️ No skills detected — try adding more detail to the JD.")
                else:
                    st.warning("Please paste a JD first.")

        default_skills = st.session_state.get(
            "auto_skills", "Python, Machine Learning, Communication"
        )
        req_skills_input = st.text_area(
            "Required Skills (comma-separated)",
            value=default_skills,
            height=90,
            placeholder="Python, SQL, Machine Learning, Communication …",
        )
        req_degrees_input = st.text_input(
            "Required Degrees",
            value="B.Tech, M.Sc, MCA",
            placeholder="B.Tech, MBA, PhD …",
        )

    # ── Right column: upload ──────────────────────────────────────────────
    with right:
        st.subheader("📁 Upload Resumes")
        uploaded_file = st.file_uploader(
            "Upload a ZIP containing PDF / DOCX resumes",
            type="zip",
            help="All .pdf and .docx files inside the ZIP will be processed.",
        )
        if uploaded_file:
            st.success(
                f"✅ `{uploaded_file.name}` · {uploaded_file.size // 1024} KB"
            )

    st.divider()
    run = st.button("🚀  Start Screening", type="primary", use_container_width=True)

    # ── Run screening ─────────────────────────────────────────────────────
    if run:
        if not uploaded_file:
            st.warning("⚠️ Please upload a ZIP file first.")
            st.stop()

        req_skills  = [s.strip() for s in req_skills_input.split(",") if s.strip()]
        req_degrees = [d.strip() for d in req_degrees_input.split(",") if d.strip()]

        if not req_skills:
            st.warning("⚠️ Enter at least one required skill.")
            st.stop()

        all_results, errors = [], []

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "resumes.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)

            files = (
                glob.glob(os.path.join(tmpdir, "**", "*.pdf"),  recursive=True)
                + glob.glob(os.path.join(tmpdir, "**", "*.docx"), recursive=True)
            )

            if not files:
                st.error("❌ No PDF or DOCX files found in the ZIP.")
                st.stop()

            st.info(f"📂 Found **{len(files)} resumes** — processing …")
            bar    = st.progress(0)
            status = st.empty()

            for i, fp in enumerate(files):
                status.text(f"Processing {os.path.basename(fp)}  ({i+1}/{len(files)})")
                try:
                    cand = screener.process_file(fp, req_skills, req_degrees)
                    all_results.append(cand)
                except Exception as e:
                    errors.append((os.path.basename(fp), str(e)))
                bar.progress((i + 1) / len(files))

            status.text("✅ Done!")

        st.session_state["results"]    = all_results
        st.session_state["req_skills"] = req_skills
        st.session_state["errors"]     = errors

    # ── Display results ───────────────────────────────────────────────────
    if "results" in st.session_state:
        all_results = st.session_state["results"]
        req_skills  = st.session_state.get("req_skills", [])
        errors      = st.session_state.get("errors", [])

        cutoff  = 0 if show_all else threshold
        results = sorted(
            [c for c in all_results if c.final_score >= cutoff],
            key=lambda x: x.final_score,
            reverse=True,
        )[:top_n]

        # ── Summary metrics ───────────────────────────────────────────────
        st.subheader("📊 Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Uploaded", len(all_results) + len(errors))
        m2.metric("Processed OK",   len(all_results))
        m3.metric("Shortlisted",    len(results))
        rejected = len(all_results) - len(results)
        m4.metric("Below Threshold", rejected)
        avg = (
            sum(c.final_score for c in results) / len(results)
            if results else 0
        )
        m5.metric("Avg Score", f"{avg:.1f}%")

        if errors:
            with st.expander(f"⚠️ {len(errors)} file(s) skipped"):
                for fname, err in errors:
                    st.markdown(f"- **{fname}**: {err}")

        # ── Download buttons ──────────────────────────────────────────────
        if results:
            dl1, dl2 = st.columns(2)

            # Excel
            df = pd.DataFrame(
                [
                    {
                        "Rank":              i + 1,
                        "Filename":          c.filename,
                        "Name":              c.name,
                        "Email":             c.email,
                        "Phone":             c.phone,
                        "Final Score (%)":   c.final_score,
                        "Skills Matched":    c.skill_score,
                        "Total Skills":      len(req_skills),
                        "Skill %":           round(c.skill_score / max(len(req_skills), 1) * 100, 1),
                        "Experience (yrs)":  c.years_experience,
                        "Exp Score":         c.experience_score,
                        "Highest Degree":    c.highest_degree,
                        "Qual Score":        c.qualification_score,
                        "Matched Skills":    ", ".join(s[0] for s in c.matched_skills),
                        "Missing Skills":    ", ".join(c.unmatched_skills),
                    }
                    for i, c in enumerate(results)
                ]
            )
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Shortlisted")
            dl1.download_button(
                "📥 Download Excel Report",
                data=buf.getvalue(),
                file_name=f"shortlisted_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            dl2.markdown("")   # spacer

        st.divider()

        # ── Candidate cards ───────────────────────────────────────────────
        if not results:
            st.warning(
                f"No candidates scored ≥ {cutoff}%. "
                "Lower the threshold in the sidebar or upload more resumes."
            )
        else:
            st.subheader(f"🏆 Shortlisted Candidates — Top {len(results)}")

            for rank, cand in enumerate(results, 1):
                score = cand.final_score
                badge_cls = (
                    "badge-high" if score >= 75
                    else "badge-med"  if score >= 55
                    else "badge-low"
                )

                with st.container(border=True):
                    h1, h2 = st.columns([4, 1])
                    h1.markdown(
                        f"**#{rank} &nbsp; {cand.filename}**"
                        f"&ensp;·&ensp; 👤 {cand.name}"
                        f"&ensp;·&ensp; 📧 {cand.email}"
                        f"&ensp;·&ensp; 📞 {cand.phone}"
                    )
                    h2.markdown(
                        f'<span class="{badge_cls}">{score:.1f}%</span>',
                        unsafe_allow_html=True,
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🎯 Final Score",    f"{cand.final_score:.1f}%")
                    c2.metric("✅ Skills Matched",  f"{cand.skill_score}/{len(req_skills)}")
                    c3.metric("💼 Experience",      f"{cand.years_experience} yr")
                    c4.metric("🎓 Degree",          cand.highest_degree)

                    with st.expander("🔍 Skill breakdown"):
                        ec1, ec2 = st.columns(2)
                        ec1.markdown("**Matched skills**")
                        pills_matched = " ".join(
                            f'<span class="pill-match">{s} &nbsp;{conf*100:.0f}%</span>'
                            for s, conf in cand.matched_skills
                        )
                        ec1.markdown(pills_matched or "_None_", unsafe_allow_html=True)

                        ec2.markdown("**Missing skills**")
                        pills_missing = " ".join(
                            f'<span class="pill-missing">{s}</span>'
                            for s in cand.unmatched_skills
                        )
                        ec2.markdown(pills_missing or "✅ All matched!", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    if "results" not in st.session_state or not st.session_state["results"]:
        st.info("▶️ Run a screening first — analytics will appear here.")
    else:
        all_results = st.session_state["results"]
        req_skills  = st.session_state.get("req_skills", [])
        cutoff      = 0 if show_all else threshold

        shortlisted = [c for c in all_results if c.final_score >= cutoff]

        st.subheader("📊 Analytics Dashboard")

        row1_l, row1_r = st.columns(2)

        # Score distribution
        scores = [c.final_score for c in all_results]
        fig1 = px.histogram(
            x=scores, nbins=12,
            title="Score Distribution (all candidates)",
            labels={"x": "Final Score (%)", "y": "Count"},
            color_discrete_sequence=["#1565C0"],
        )
        fig1.add_vline(
            x=threshold, line_dash="dash", line_color="#e53935",
            annotation_text=f"Threshold ({threshold}%)",
            annotation_position="top right",
        )
        fig1.update_layout(showlegend=False)
        row1_l.plotly_chart(fig1, use_container_width=True)

        # Top-N bar chart
        top_n_sorted = sorted(all_results, key=lambda c: c.final_score, reverse=True)[:15]
        fig2 = px.bar(
            x=[c.filename[:22] + "…" if len(c.filename) > 22 else c.filename
               for c in top_n_sorted],
            y=[c.final_score for c in top_n_sorted],
            title="Top 15 Candidates by Final Score",
            labels={"x": "Resume", "y": "Score (%)"},
            color=[c.final_score for c in top_n_sorted],
            color_continuous_scale="RdYlGn",
        )
        fig2.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        row1_r.plotly_chart(fig2, use_container_width=True)

        row2_l, row2_r = st.columns(2)

        # Skill gap — how many candidates have each skill
        if req_skills:
            counts = {skill: 0 for skill in req_skills}
            for cand in all_results:
                for skill, _ in cand.matched_skills:
                    if skill in counts:
                        counts[skill] += 1

            fig3 = px.bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                title="Skill Coverage Across All Candidates",
                labels={"x": "Skill", "y": "Candidates with Skill"},
                color=list(counts.values()),
                color_continuous_scale="Blues",
            )
            fig3.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
            row2_l.plotly_chart(fig3, use_container_width=True)

        # Degree distribution pie
        degrees = [c.highest_degree for c in all_results]
        fig4 = px.pie(
            names=degrees,
            title="Degree Distribution",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        row2_r.plotly_chart(fig4, use_container_width=True)

        # Score component breakdown (shortlisted only)
        if shortlisted:
            st.subheader("Score Breakdown — Shortlisted Candidates")
            top20 = sorted(shortlisted, key=lambda c: c.final_score, reverse=True)[:20]
            names = [c.filename[:18] for c in top20]
            fig5 = px.bar(
                x=names,
                y=[
                    [c.skill_score / max(len(req_skills), 1) * 100 * 0.55 for c in top20],
                    [c.experience_score * 0.25 for c in top20],
                    [c.qualification_score * 0.20 for c in top20],
                ],
                title="Stacked Score Components (top 20 shortlisted)",
                labels={"x": "Resume", "value": "Score contribution"},
                barmode="stack",
                color_discrete_sequence=["#1976D2", "#43A047", "#FB8C00"],
            )
            new_names = {"wide_variable_0": "Skills (55%)",
                         "wide_variable_1": "Experience (25%)",
                         "wide_variable_2": "Qualification (20%)"}
            fig5.for_each_trace(lambda t: t.update(name=new_names.get(t.name, t.name)))
            fig5.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig5, use_container_width=True)
