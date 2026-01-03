from domain.candidate import Candidate

class ScreenerService:
    def __init__(self, file_reader, skill_matcher, parser_service):
        self.reader = file_reader
        self.skill_matcher = skill_matcher
        self.parser = parser_service

    def process_file(self, file_path, req_skills, req_degrees):
        filename = file_path.split("/")[-1].split("\\")[-1] # Robust split
        
        # 1. Extract Text
        text = self.reader.extract_text(file_path)
        if not text:
            raise Exception("Empty or unreadable file")

        # 2. Create Candidate
        candidate = Candidate(filename, text)

        # 3. Analyze Skills
        matched, score_count = self.skill_matcher.evaluate_skills(text, req_skills)
        candidate.matched_skills = matched
        candidate.skill_score = score_count

        # 4. Analyze Exp & Edu
        candidate.experience_score = self.parser.calculate_experience_score(text)
        candidate.qualification_score = self.parser.calculate_qualification_score(text, req_degrees)

        # 5. Finalize
        candidate.calculate_final_score(len(req_skills))
        
        return candidate