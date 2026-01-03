from nltk import ngrams
from fuzzywuzzy import fuzz

class SkillMatcher:
    def __init__(self, ml_engine):
        self.ml_engine = ml_engine

    def _generate_phrases(self, text, n=2):
        tokens = text.lower().split()
        return [' '.join(gram) for gram in ngrams(tokens, n)]

    def match_single_skill(self, required_skill, resume_text, threshold=0.85, fuzzy_threshold=85):
        # 1. Vector Matching (Semantic)
        skill_vec = self.ml_engine.get_vector(required_skill)
        
        # Prepare Resume Phrases (Bigrams & Trigrams)
        phrases = self._generate_phrases(resume_text, 2) + self._generate_phrases(resume_text, 3)
        
        if skill_vec is not None:
            for phrase in phrases:
                phrase_vec = self.ml_engine.get_vector(phrase)
                similarity = self.ml_engine.cosine_similarity(skill_vec, phrase_vec)
                if similarity > threshold:
                    return True, similarity  # Match found via AI

        # 2. Fuzzy Fallback (Spelling/Exact)
        # Check against phrases first
        for phrase in phrases:
            if fuzz.partial_ratio(required_skill.lower(), phrase) >= fuzzy_threshold:
                return True, 1.0
        
        # Last resort: Simple substring check
        if required_skill.lower() in resume_text.lower():
             return True, 1.0

        return False, 0.0

    def evaluate_skills(self, text, required_skills):
        matched_list = []
        score_count = 0
        
        for skill in required_skills:
            skill = skill.strip()
            if not skill: continue
            
            is_match, score = self.match_single_skill(skill, text)
            if is_match:
                matched_list.append((skill, score))
                score_count += 1
                
        return matched_list, score_count