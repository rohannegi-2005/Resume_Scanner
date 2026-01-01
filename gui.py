import streamlit as st
import os
import zipfile
import tempfile
import glob
from resume_logic import (
    extract_experience,
    match_qualification,
    calculate_final_score   
)

from text_extractor import TextExtractor
from semantic_matching import SemanticMatcher
from text_vectorizer import TextVectorizer

text_extractor = TextExtractor()
semantic_matcher = SemanticMatcher()
text_vectorizer = TextVectorizer()
# App title
st.set_page_config(page_title="Resume Screening Tool", layout="wide")
st.title("📄 AI Resume Screening Tool")
st.markdown("Upload Resumes (ZIP), Set Skills & Degrees, Get Matched Candidates")

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

        st.info("⏳ Processing Resumes...")
        progress = st.progress(0)
        total_files = len(resume_files)

        for i, file_path in enumerate(resume_files, 1):
            filename = os.path.basename(file_path)
            try:
                text = text_extractor.extract_text(file_path)
                tokens = text_extractor.get_tokens(text)
                bigram_phrases = text_extractor.get_phrases(tokens, 2)
                trigram_phrases = text_extractor.get_phrases(tokens, 3)
                phrases = bigram_phrases + trigram_phrases
                phrase_vectors = text_vectorizer.word_to_vec(phrases)
                skill_vectors = text_vectorizer.word_to_vec(required_skills)  # Precompute vectors if needed
                matched_skills, skill_score = semantic_matcher.check_match(
                    skill_vectors,
                    phrase_vectors,
                    threshold=0.85
                )

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
