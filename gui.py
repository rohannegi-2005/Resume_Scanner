import os
import requests
from gensim.models import KeyedVectors

def download_model(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading: {dest_path}")
        r = requests.get(url, stream=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {dest_path}")

# Model file paths
os.makedirs('glove_model', exist_ok=True)
MODEL_KV_PATH = 'glove_model/glove_model.kv'
MODEL_NPY_PATH = 'glove_model/glove_model.kv.vectors.npy'

# 🔗 URLs to both files (replace with your actual GitHub release links if different)
KV_URL = 'https://github.com/rohannegi-2005/Resume_Scanner/releases/download/version_1/glove_model.kv'
NPY_URL = 'https://github.com/rohannegi-2005/Resume_Scanner/releases/download/version_1/glove_model.kv.vectors.npy'

# Download both files
download_model(KV_URL, MODEL_KV_PATH)
download_model(NPY_URL, MODEL_NPY_PATH)

# Load model
model = KeyedVectors.load(MODEL_KV_PATH)


import streamlit as st
import os
import zipfile
import tempfile
import glob
from resume_logic import (
    extract_text,
    match_skills,
    extract_experience,
    match_qualification,
    calculate_final_score,
    word2vec,
)

# App title
st.set_page_config(page_title="Resume Screening Tool", layout="wide")
st.title("📄 AI Resume Screening Tool")
st.markdown("Upload a folder (zip) with resumes, specify required skills and degrees, and get candidates with a score above 60%.")

# Input for required skills and degrees
required_skills = st.text_input("✏️ Required Skills (comma-separated)", value="")
required_degrees = st.text_input("🎓 Required Degrees (comma-separated)", value="")

# Upload zip of resumes
uploaded_file = st.file_uploader("📁 Upload Resume Folder (ZIP only)", type=["zip"])

# Process resumes
if uploaded_file and required_skills and required_degrees:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, uploaded_file.name)

        # Save uploaded zip
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract all contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Recursively find all PDF and DOCX files
        resume_files = glob.glob(os.path.join(tmpdir, "**", "*.pdf"), recursive=True)
        resume_files += glob.glob(os.path.join(tmpdir, "**", "*.docx"), recursive=True)

        selected_resumes = []

        st.info("⏳ Processing resumes...")
        progress = st.progress(0)
        total_files = len(resume_files)

        for i, file_path in enumerate(resume_files, 1):
            filename = os.path.basename(file_path)
            try:
                text = extract_text(file_path)
                matched_skills, skill_score = match_skills(text, required_skills.split(","), word2vec)
                skill_score = (skill_score / len(required_skills.split(","))) * 100

                exp_score = extract_experience(text)
                qual_score = match_qualification(text, required_degrees.split(","))
                final_score = calculate_final_score(skill_score, exp_score, qual_score)

                if final_score >= 60:
                    selected_resumes.append((filename, round(final_score, 2), matched_skills))
            except Exception as e:
                st.warning(f"❌ Could not process {filename}: {e}")

            progress.progress(i / total_files)

        # Output
        st.success("✅ Resume Processing Complete!")

        if selected_resumes:
            st.subheader("🎯 Selected Resumes")
            for name, score, skills in selected_resumes:
                st.markdown(f"**📄 {name}** — Final Score: **{score}%**")
                st.markdown(f"🔧 Matched Skills: `{', '.join([s[0] for s in skills])}`")
                st.markdown("---")
        else:
            st.warning("⚠️ No resumes met the criteria.")
