from text_similarity_engine import TextSimilarityEngine

class SemanticMatcher:
    def __init__(self):
        self.similarity_engine = TextSimilarityEngine()

    def check_match(self, skills, phrases, threshold=0.85):
        matched_skills = []
        skill_score = 0

        for skill in skills:
            similarity = self.similarity_engine.similarity(skill, phrases)
            if similarity >= threshold:
                matched_skills.append((skill, similarity))
                skill_score += 1

        return matched_skills, skill_score
    