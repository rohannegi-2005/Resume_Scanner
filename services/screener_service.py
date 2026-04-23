"""
ScreenerService: Orchestrates the full resume processing pipeline.

Scoring formula (weights can be tuned via class constants):
  Final = (skill_pct × 0.55) + (exp_score × 0.25) + (qual_score × 0.20)

Experience scoring: 0–10 years maps linearly to 0–100.
Qualification scoring:
  - If candidate degree matches required degrees list → 100
  - Otherwise uses degree rank table as partial credit
"""
import os
from domain.candidate import Candidate
from infrastructure.file_reader import FileReader
from services.skill_matcher import SkillMatcher
from services.parser_service import ParserService


class ScreenerService:
    # ── Scoring weights (must sum to 1.0) ─────────────────────────────
    SKILL_WEIGHT = 0.55
    EXPERIENCE_WEIGHT = 0.25
    QUALIFICATION_WEIGHT = 0.20

    # ── Degree → partial credit score ─────────────────────────────────
    DEGREE_SCORES = {
        "PhD": 100, "M.Tech": 95, "M.Sc": 90, "MBA": 90,
        "MCA": 85,  "B.Tech": 80, "B.Sc": 75, "BCA": 70,
        "B.Com": 65, "B.A": 60,  "12th": 30,  "10th": 15,
        "Unknown": 10,
    }

    def __init__(
        self,
        file_reader: FileReader,
        skill_matcher: SkillMatcher,
        parser_service: ParserService,
    ):
        self.file_reader    = file_reader
        self.skill_matcher  = skill_matcher
        self.parser_service = parser_service

    def process_file(
        self,
        file_path: str,
        req_skills: list,
        req_degrees: list,
    ) -> Candidate:
        # 1. Read raw text
        text = self.file_reader.read(file_path)
        if not text.strip():
            raise ValueError("Resume appears empty or unreadable.")

        # 2. Parse contact info
        name, email, phone = self.parser_service.extract_contact(text)

        # 3. Skill matching (3-layer)
        matched_skills, unmatched_skills = self.skill_matcher.match_skills(
            text, req_skills
        )

        # 4. Experience score (0–100, capped at 10 years = 100)
        years_exp = self.parser_service.extract_experience(text)
        exp_score = min(years_exp * 10.0, 100.0)

        # 5. Qualification score
        highest_degree = self.parser_service.extract_highest_degree(text)
        qual_score = self._score_qualification(highest_degree, req_degrees)

        # 6. Skill percentage
        skill_pct = (
            len(matched_skills) / len(req_skills) * 100.0
            if req_skills else 0.0
        )

        # 7. Weighted final score
        final_score = (
            skill_pct   * self.SKILL_WEIGHT
            + exp_score * self.EXPERIENCE_WEIGHT
            + qual_score * self.QUALIFICATION_WEIGHT
        )

        return Candidate(
            filename           = os.path.basename(file_path),
            name               = name,
            email              = email,
            phone              = phone,
            final_score        = round(final_score, 1),
            skill_score        = len(matched_skills),
            experience_score   = round(exp_score, 1),
            qualification_score= round(qual_score, 1),
            matched_skills     = matched_skills,
            unmatched_skills   = unmatched_skills,
            years_experience   = round(years_exp, 1),
            highest_degree     = highest_degree,
            file_path          = file_path,
        )

    # ── Private ───────────────────────────────────────────────────────

    def _score_qualification(self, highest_degree: str, req_degrees: list) -> float:
        base = self.DEGREE_SCORES.get(highest_degree, 10)
        if not req_degrees:
            return float(base)

        # Normalise comparison (case-insensitive, strip dots/spaces)
        def normalise(d: str) -> str:
            return d.lower().replace(".", "").replace(" ", "")

        req_normalised = {normalise(d) for d in req_degrees}
        if normalise(highest_degree) in req_normalised:
            return 100.0

        return float(base)
