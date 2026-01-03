class Candidate:
    def __init__(self, filename, text):
        self.filename = filename
        self.text = text
        self.matched_skills = []  # List of (skill, score)
        self.skill_score = 0.0
        self.experience_score = 0.0
        self.qualification_score = 0.0
        self.final_score = 0.0

    def calculate_final_score(self, total_req_skills):
        # Normalize skill score (percentage of skills matched)
        normalized_skill_score = (self.skill_score / total_req_skills) * 100 if total_req_skills > 0 else 0
        
        # Weighted Formula: 60% Skills + 30% Exp + 10% Qual
        self.final_score = (
            (normalized_skill_score * 0.6) + 
            (self.experience_score * 0.3) + 
            (self.qualification_score * 0.1)
        )
        return self.final_score