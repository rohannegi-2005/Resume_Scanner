import math

class SemanticMatcher:
    
    def _cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


    def check_match(self, skills, phrases, threshold=0.85):
        
        matched_skills = []
        skill_score = 0

        for skill in skills:
            for phrase in phrases:
                similarity = self._cosine_similarity(skill, phrase)
                if similarity >= threshold:
                    matched_skills.append((skill, similarity))
                    skill_score += 1

        return matched_skills, skill_score
    