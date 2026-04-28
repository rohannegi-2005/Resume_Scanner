"""
ScreenerService v3 — SaaS-grade scoring

Scoring formula (v3 changes):
  ─────────────────────────────────────────────────────────────────
  SKILLS      → 55 %   (unchanged — most predictive signal)
  EXPERIENCE  → 25 %   (with FRESHER BASELINE: student ≥ 10 pts)
  QUALIFICATION → 20 % (now normalised against required degrees)
  ─────────────────────────────────────────────────────────────────

Fresher baseline:
  If years_exp == 0 AND candidate is a student (B.Tech/B.Sc/etc.)
  we award a 10-pt baseline so a strong B.Tech fresher with all
  skills isn't unfairly crushed by 0 experience.

Qualification scoring:
  - Exact match to required degrees   → 100
  - Higher degree than required        → 100
  - Lower degree than required         → partial credit (50–85)
  - Unknown                            → 10
"""
import os
from domain.candidate import Candidate
from infrastructure.file_reader import FileReader
from services.skill_matcher import SkillMatcher
from services.parser_service import ParserService


class ScreenerService:

    # ── Scoring weights (must sum to 1.0) ─────────────────────────────
    SKILL_WEIGHT        = 0.55
    EXPERIENCE_WEIGHT   = 0.25
    QUALIFICATION_WEIGHT = 0.20

    # ── Degree rank (higher index = higher qualification) ──────────────
    _DEGREE_ORDER = [
        "10th", "12th", "Diploma", "B.A", "B.Com", "BBA",
        "BCA", "B.Sc", "B.Tech", "MCA", "MBA", "M.Sc", "M.Tech", "PhD",
    ]

    # ── Flat score for each degree when no required-degree list given ──
    _DEGREE_BASE = {
        "PhD": 100, "M.Tech": 95, "M.Sc": 90, "MBA": 90,
        "MCA": 85,  "B.Tech": 80, "B.Sc": 75, "BCA": 70,
        "BBA": 68,  "B.Com": 65, "B.A": 60,  "Diploma": 45,
        "12th": 30, "10th": 15,  "Unknown": 10,
    }

    # Degrees considered "student / fresher" for baseline bonus
    _STUDENT_DEGREES = {
        "B.Tech", "B.Sc", "BCA", "BBA", "B.Com", "B.A", "Diploma",
        "MCA", "MBA", "M.Tech", "M.Sc",
    }

    def __init__(
        self,
        file_reader:    FileReader,
        skill_matcher:  SkillMatcher,
        parser_service: ParserService,
    ):
        self.file_reader    = file_reader
        self.skill_matcher  = skill_matcher
        self.parser_service = parser_service

    # ─────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────

    def process_file(
        self,
        file_path:   str,
        req_skills:  list,
        req_degrees: list,
    ) -> Candidate:

        # 1. Read
        text = self.file_reader.read(file_path)
        if not text.strip():
            raise ValueError("Resume appears empty or unreadable.")

        # 2. Parse contact
        name, email, phone = self.parser_service.extract_contact(text)

        # 3. Skill matching (3-layer: exact → fuzzy → semantic)
        matched_skills, unmatched_skills = self.skill_matcher.match_skills(
            text, req_skills
        )

        # 4. Experience
        years_exp = self.parser_service.extract_experience(text)
        highest_degree = self.parser_service.extract_highest_degree(text)

        # 5. Scores
        skill_pct   = (len(matched_skills) / len(req_skills) * 100.0
                       if req_skills else 0.0)
        exp_score   = self._score_experience(years_exp, highest_degree)
        qual_score  = self._score_qualification(highest_degree, req_degrees)

        # 6. Weighted final
        final_score = (
            skill_pct   * self.SKILL_WEIGHT
            + exp_score * self.EXPERIENCE_WEIGHT
            + qual_score * self.QUALIFICATION_WEIGHT
        )

        return Candidate(
            filename            = os.path.basename(file_path),
            name                = name,
            email               = email,
            phone               = phone,
            final_score         = round(final_score, 1),
            skill_score         = len(matched_skills),
            experience_score    = round(exp_score, 1),
            qualification_score = round(qual_score, 1),
            matched_skills      = matched_skills,
            unmatched_skills    = unmatched_skills,
            years_experience    = round(years_exp, 1),
            highest_degree      = highest_degree,
            file_path           = file_path,
        )

    # ─────────────────────────────────────────────────────────────────
    # Private — scoring helpers
    # ─────────────────────────────────────────────────────────────────

    def _score_experience(self, years: float, degree: str) -> float:
        """
        Map years of experience to 0–100.
        Cap at 10 years = 100.
        Fresher baseline: student with 0 years → 10 pts (not 0).
        """
        if years == 0.0 and degree in self._STUDENT_DEGREES:
            return 10.0                       # fresher baseline
        return min(years * 10.0, 100.0)

    def _score_qualification(
        self, highest_degree: str, req_degrees: list
    ) -> float:
        """
        Score 0–100 based on how well the candidate's degree
        matches the required degrees list.

        Logic:
          • If req_degrees is empty → use flat DEGREE_BASE score.
          • If exact match (case-insensitive, ignoring dots/spaces) → 100.
          • If candidate has a HIGHER degree than any required → 100.
          • If candidate has a LOWER degree → partial credit (50–85 range).
        """
        base = float(self._DEGREE_BASE.get(highest_degree, 10))

        if not req_degrees:
            return base

        # Normalise for comparison
        def norm(d: str) -> str:
            return d.lower().replace(".", "").replace(" ", "")

        req_normed = {norm(d) for d in req_degrees}
        cand_normed = norm(highest_degree)

        # Exact match
        if cand_normed in req_normed:
            return 100.0

        # Compare ranks
        cand_rank = self._degree_rank(highest_degree)
        req_ranks = [self._degree_rank(d) for d in req_degrees]
        max_req   = max(req_ranks) if req_ranks else 0

        if cand_rank >= max_req:
            return 100.0            # overqualified → full marks

        # Partial credit: scale base score up slightly
        # (so a B.Sc when B.Tech required still gets ~70, not just base)
        if cand_rank > 0 and max_req > 0:
            ratio = cand_rank / max_req
            partial = 50.0 + ratio * 50.0
            return round(min(partial, 99.0), 1)

        return base

    def _degree_rank(self, degree: str) -> int:
        try:
            return self._DEGREE_ORDER.index(degree) + 1
        except ValueError:
            return 0
