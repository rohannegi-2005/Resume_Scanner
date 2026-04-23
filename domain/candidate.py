from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Candidate:
    filename: str = ""
    name: str = ""
    email: str = ""
    phone: str = ""
    final_score: float = 0.0
    skill_score: int = 0
    experience_score: float = 0.0
    qualification_score: float = 0.0
    matched_skills: List[Tuple[str, float]] = field(default_factory=list)
    unmatched_skills: List[str] = field(default_factory=list)
    years_experience: float = 0.0
    highest_degree: str = "Unknown"
    file_path: str = ""
