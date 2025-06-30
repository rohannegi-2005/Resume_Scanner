#!/usr/bin/env python
# coding: utf-8

# Import Libraries

import os
import pdfplumber
import docx
import re
from fuzzywuzzy import fuzz



# ========== CONFIG ==========
RESUME_FOLDER = "resumes"
# REQUIRED_SKILLS = ["game engine", "3D Modeling Basics", "c#","animation systems", "unity","ai","c++","java"]
# REQUIRED_DEGREES = ["B.tech","Engineering","Diploma"]



# # Function To Extract Text From PDF or DOCX

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""



# #  Experience Checking


from dateutil import parser
from datetime import datetime
import re

def extract_experience(text):
    text = text.lower()
    total_months = 0

    # STEP 1: Extract only from "experience" section
    experience_section = ""

    # Look for heading keywords
    experience_headings = ["experience", "work experience", "professional experience", "employment history"]
    lines = text.split("\n")

    capturing = False
    for line in lines:
        if any(heading in line for heading in experience_headings):
            capturing = True
            continue
        elif capturing and (line.strip() == "" or line.strip().endswith(":")):
            # End of section if new heading or blank line
            break
        elif capturing:
            experience_section += line + "\n"

    # STEP 2: Extract date ranges only from experience_section
    patterns = [
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*[-to]+\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|present)',   # 06/07/2022 - 06/05/2023
        r'(\d{1,2}[-/]\d{2,4})\s*[-to]+\s*(\d{1,2}[-/]\d{2,4}|present)',                         # 06/2022 - 05/2023
        r'([a-z]+\s\d{4})\s*[-to]+\s*([a-z]+\s\d{4}|present)'                                    # June 2022 - July 2023
    ]

    for pattern in patterns:
        matches = re.findall(pattern, experience_section)
        for start_str, end_str in matches:
            try:
                start_date = parser.parse(start_str, dayfirst=True)
                end_date = parser.parse(end_str, dayfirst=True) if "present" not in end_str else datetime.now()
                months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                total_months += max(0, months)
            except:
                continue

    # STEP 3: Convert total experience to score
    if total_months >= 60:
        return 100
    elif total_months >= 36:
        return 70
    elif total_months >= 24:
        return 50
    elif total_months >= 12:
        return 25
    else:
        return 0





# #  Qualification Matching

def extract_education_section(text):
    # Normalize text
    lines = text.lower().split('\n')

    edu_keywords = ['education', 'educational qualification', 'academic background', 'qualification']
    section_lines = []
    found = False

    for line in lines:
        if any(keyword in line for keyword in edu_keywords):
            found = True
            continue

        if found:
            if line.strip() == "" or line.strip().endswith(":"):  # stop if empty or new section starts
                break
            section_lines.append(line)

    return " ".join(section_lines)


def match_qualification(text, degrees):
    edu_section = extract_education_section(text)
    for degree in degrees:
        if degree.lower() in edu_section:
            return 100
    return 0



#   Score Calculation

def calculate_final_score(skill_score, exp_score, qual_score):
    return ((skill_score * 0.6) + (exp_score * 0.3) + (qual_score * 0.1))


#                                  ''' WORD EMBEDDING '''
# --------------------------------------------------------------------------------------------------------
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.models import KeyedVectors

# glove_input_file = "D:\\google\\glove.twitter.27B.100d.txt"
# word2vec_output_file = "D:\\google\\glove.twitter.27B.100d.word2vec.txt"
# model_output_file = "D:\\google\\glove_model.kv"

# # Convert and load once
# glove2word2vec(glove_input_file, word2vec_output_file)
# word2vec = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# # Save the processed model (compact, loads faster)
# word2vec.save(model_output_file)
# ----------------------------------------------------------------------------------------------------------


from gensim.models import KeyedVectors
import os

# Load pre-saved GloVe model in .kv format

model_path = os.path.join("glove_model", "glove_model.kv")
word2vec = KeyedVectors.load(model_path, mmap='r')  # mmap='r' is optional but helps with memory efficiency



import numpy as np

def get_average_vector(text, model):
    words = text.lower().split()
    valid_vectors = []

    for word in words:
        if word in model:
            valid_vectors.append(model[word])

    if valid_vectors:
        return np.mean(valid_vectors, axis=0)
    else:
        return None

# Cosine Similarirty

from numpy import dot
from numpy.linalg import norm
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))




from nltk import ngrams
def generate_phrases(tokens, n=2):
    return [' '.join(gram) for gram in ngrams(tokens, n)]

#  Skill Matching Using Word2Vec and Fuzzzywords

def match_skill_with_resume(required_skill, resume_text, model, threshold=0.85, fuzzy_threshold=85):
    tokens = resume_text.lower().split()
    
    # Create phrases
    bigrams = generate_phrases(tokens, 2)
    trigrams = generate_phrases(tokens, 3)
    phrases = bigrams + trigrams

    skill_vector = get_average_vector(required_skill, model)
    
    top_score = 0
    matched = False

    if skill_vector is not None:
        for phrase in phrases:
            phrase_vector = get_average_vector(phrase, model)
            if phrase_vector is not None:
                similarity = cosine_similarity(skill_vector, phrase_vector)
                if similarity > threshold:
                    matched = True
                    top_score = max(top_score, similarity)

        if matched:
            return True, top_score

    # 🔁 Fuzzy fallback if not matched via vectors
    for phrase in phrases:
        fuzzy_score = fuzz.partial_ratio(required_skill.lower(), phrase.lower())
        if fuzzy_score >= fuzzy_threshold:
            return True, fuzzy_score / 100  # normalize to 0–1

    return False, 0


def match_skills(full_text, required_skills, model, threshold=0.85):
    matched_skills = []
    
    skill_score = 0
 
    for skill in required_skills:
        matched, similarity_score = match_skill_with_resume(skill, full_text, model, threshold)
        if matched:
            matched_skills.append((skill, similarity_score))
            skill_score += 1

    return matched_skills, skill_score
        




                               # TESTING

# for filename in os.listdir(RESUME_FOLDER):
#     file_path = os.path.join(RESUME_FOLDER, filename)
#     if not filename.endswith((".pdf", ".docx")):
#         continue

#     print(f"\n📄 Analyzing Resume: {filename}")
#     text = extract_text(file_path)
    

#     matched_skills, skill_score = match_skills(text, REQUIRED_SKILLS, word2vec)
#     total_required_skills = len(REQUIRED_SKILLS)
#     skill_score = (skill_score / total_required_skills) * 100

#     exp_score = extract_experience(text)
#     qual_score = match_qualification(text, REQUIRED_DEGREES)
#     final_score = calculate_final_score(skill_score, exp_score, qual_score)
    


    
#     print(f"✅ Matched Skills: {matched_skills}")
#     print(f"🎯 Skill Score: {skill_score}%")
#     print(f"💼 Experience Score: {exp_score}%")
    
#     print(f"🎓 Qualification Score: {qual_score}%")
    
#     print(f"📊 Final Score: {final_score}%")
#     print("-" * 60)
    



