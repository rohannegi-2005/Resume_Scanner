"""
ParserService:
  - extract_contact()        → (name, email, phone)
  - extract_experience()     → float years
  - extract_highest_degree() → str degree label
  - extract_skills_from_jd() → list of skill strings from a pasted JD
"""
import re
from typing import List, Tuple

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class ParserService:
    # ── Regex patterns ────────────────────────────────────────────────
    EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")
    PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
    EXP_RE = re.compile(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s*(?:work\s+)?(?:experience|exp)",
        re.IGNORECASE,
    )

    # ── Degree taxonomy (ordered highest → lowest) ────────────────────
    DEGREE_RANK = [
        "PhD", "M.Tech", "M.Sc", "MBA", "MCA",
        "B.Tech", "B.Sc", "BCA", "B.Com", "B.A", "12th", "10th",
    ]
    DEGREE_KEYWORDS = {
        "PhD":    ["ph.d", "phd", "doctor of philosophy", "doctorate"],
        "M.Tech": ["m.tech", "master of technology", "m tech", "mtech"],
        "M.Sc":   ["m.sc", "master of science", "m sc", "msc"],
        "MBA":    ["mba", "master of business administration", "master of business"],
        "MCA":    ["mca", "master of computer application"],
        "B.Tech": ["b.tech", "bachelor of technology", "b tech", "btech",
                   "b.e.", "b.e", "bachelor of engineering", "be "],
        "B.Sc":   ["b.sc", "bachelor of science", "bsc", "b sc"],
        "BCA":    ["bca", "bachelor of computer application"],
        "B.Com":  ["b.com", "bachelor of commerce", "bcom"],
        "B.A":    ["b.a.", " b.a ", "bachelor of arts"],
        "12th":   ["12th", "hsc", "higher secondary", "class xii", "intermediate"],
        "10th":   ["10th", "ssc", "matriculation", "class x", "class 10"],
    }

    # ── Common tech skills for JD parsing ─────────────────────────────
    TECH_PATTERN = re.compile(
        r"\b(Python|Java(?:Script)?|TypeScript|React(?:\.js)?|Vue(?:\.js)?|Angular|"
        r"Node(?:\.js)?|Django|Flask|FastAPI|Spring Boot|"
        r"SQL|MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|"
        r"Docker|Kubernetes|AWS|GCP|Azure|CI/?CD|Terraform|"
        r"TensorFlow|PyTorch|scikit-?learn|Keras|"
        r"Machine Learning|Deep Learning|NLP|Computer Vision|"
        r"Data Analysis|Pandas|NumPy|Spark|Hadoop|"
        r"Power BI|Tableau|Excel|Looker|"
        r"Git|REST(?:ful)? API|GraphQL|gRPC|Kafka|RabbitMQ|"
        r"HTML5?|CSS3?|Sass|Tailwind|"
        r"C\+\+|C#|Go|Rust|Kotlin|Swift|Flutter|React Native|"
        r"Linux|Bash|Shell|Agile|Scrum|Jira|"
        r"Communication|Teamwork|Leadership|Problem[- ]Solving|"
        r"Microservices|DevOps|MLOps|LLM|OpenAI|LangChain)\b",
        re.IGNORECASE,
    )

    # ── Public API ────────────────────────────────────────────────────

    def extract_contact(self, text: str) -> Tuple[str, str, str]:
        email = self._extract_email(text)
        phone = self._extract_phone(text)
        name  = self._extract_name(text)
        return name, email, phone

    def extract_experience(self, text: str) -> float:
        """Return years of experience; uses year-count heuristic as fallback."""
        matches = self.EXP_RE.findall(text)
        if matches:
            return max(float(m) for m in matches)

        # Heuristic: count distinct 4-digit years (e.g. "2019 – 2022")
        years_found = set(re.findall(r"\b(20\d{2}|19\d{2})\b", text))
        if years_found:
            years_int = sorted(int(y) for y in years_found)
            span = years_int[-1] - years_int[0]
            return min(float(span), 15.0)
        return 0.0

    def extract_highest_degree(self, text: str) -> str:
        text_lower = text.lower()
        for degree in self.DEGREE_RANK:
            for keyword in self.DEGREE_KEYWORDS[degree]:
                if keyword in text_lower:
                    return degree
        return "Unknown"

    def extract_skills_from_jd(self, jd_text: str) -> List[str]:
        """
        Auto-extract skills from a pasted Job Description.
        Combines:
          1. Structured section parsing (Skills / Requirements blocks)
          2. Known tech-keyword regex sweep
        """
        skills: set = set()

        # 1. Try to find a skills/requirements section and parse bullet items
        section_re = re.compile(
            r"(?:required\s+)?(?:skills?|requirements?|qualifications?|"
            r"technologies?|tools?|must[- ]have)[:\s]+(.+?)(?=\n\s*\n|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        for block in section_re.findall(jd_text):
            for item in re.split(r"[,\n•\-\*\u2022]", block):
                item = item.strip().strip(".")
                if 2 < len(item) < 60 and not item[0].isdigit():
                    skills.add(item.title() if item.islower() else item)

        # 2. Sweep entire JD for known tech keywords
        for m in self.TECH_PATTERN.finditer(jd_text):
            skills.add(m.group())

        return sorted(skills)[:30]  # cap at 30

    # ── Private helpers ───────────────────────────────────────────────

    def _extract_email(self, text: str) -> str:
        m = self.EMAIL_RE.search(text)
        return m.group() if m else "Not Found"

    def _extract_phone(self, text: str) -> str:
        for m in self.PHONE_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 8 <= len(digits) <= 15:
                return m.group().strip()
        return "Not Found"

    def _extract_name(self, text: str) -> str:
        """
        Heuristic: scan first 8 non-empty lines for a 2–4 word all-alpha string
        that doesn't look like a section header or address.
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        IGNORE = {"resume", "curriculum vitae", "cv", "profile", "summary",
                   "objective", "contact", "address", "experience", "education"}
        for line in lines[:8]:
            if line.lower() in IGNORE:
                continue
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(re.match(r"^[A-Za-z.'\-]+$", w) for w in words):
                    if "@" not in line and not re.search(r"\d{4}", line):
                        return line.title()
        return "Unknown"
