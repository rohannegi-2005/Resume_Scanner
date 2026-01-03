import streamlit as st
import os
import zipfile
import tempfile
import glob

# Import OOP Components
from infrastructure.ml_engine import MLEngine
from infrastructure.file_reader import FileReader
from services.skill_matcher import SkillMatcher
from services.parser_service import ParserService
from services.screener_service import ScreenerService

# --- Configuration ---
st.set_page_config(page_title="AI Resume Screening", layout="wide")
MODEL_PATH = os.path.join("glove_model", "glove_model.kv")

# --- Dependency Injection ---
# 1. Initialize Infrastructure
try:
    ml_engine = MLEngine(MODEL_PATH)
    file_reader = FileReader()
except Exception as e:
    st.error(f"Setup Error: {e}. Please ensure glove_model folder exists.")
    st.stop()

# 2. Initialize Services
skill_matcher = SkillMatcher(ml_engine)
parser_service = ParserService()
screener = ScreenerService(file_reader, skill_matcher, parser_service)

# --- UI Logic ---
st.title("📄 OOP AI Resume Screening Tool")
st.markdown("Upload a ZIP of resumes. We use **Word2Vec** for semantic skill matching.")

c1, c2 = st.columns(2)
req_skills_input = c1.text_input("Required Skills (comma-separated)", "python, machine learning, communication")
req_degrees_input = c2.text_input("Required Degrees", "B.Tech, M.Sc, MCA")

uploaded_file = st.file_uploader("📁 Upload Resumes (ZIP)", type="zip")

if st.button("🚀 Start Screening") and uploaded_file:
    req_skills = [s.strip() for s in req_skills_input.split(",") if s.strip()]
    req_degrees = [d.strip() for d in req_degrees_input.split(",") if d.strip()]

    results = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Unzip
        zip_path = os.path.join(tmpdir, "resumes.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Gather Files
        files = glob.glob(os.path.join(tmpdir, "**", "*.pdf"), recursive=True)
        files += glob.glob(os.path.join(tmpdir, "**", "*.docx"), recursive=True)

        progress_bar = st.progress(0)
        
        for i, file_path in enumerate(files):
            try:
                # The Service does all the heavy lifting
                candidate = screener.process_file(file_path, req_skills, req_degrees)
                
                if candidate.final_score >= 60:
                    results.append(candidate)
            except Exception as e:
                # st.warning(f"Skipped {os.path.basename(file_path)}: {e}")
                pass
            
            progress_bar.progress((i + 1) / len(files))

    # --- Display Results ---
    st.success("Analysis Complete!")
    
    if not results:
        st.warning("No candidates met the 60% threshold.")
    
    # Sort by score high to low
    results.sort(key=lambda x: x.final_score, reverse=True)

    for cand in results:
        with st.container():
            st.subheader(f"👤 {cand.filename}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Score", f"{cand.final_score:.1f}%")
            c2.metric("Skill Match", f"{cand.skill_score}/{len(req_skills)}")
            c3.metric("Exp Score", f"{cand.experience_score}")
            c4.metric("Qual Score", f"{cand.qualification_score}")
            
            with st.expander("View Details"):
                st.write("**Matched Skills (Semantic & Fuzzy):**")
                # Show list of matched skills
                st.write([s[0] for s in cand.matched_skills])
            st.divider()