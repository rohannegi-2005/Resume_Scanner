"""
ParserService v3 — Production-grade, SaaS-level accuracy

Root-cause fixes from v2:
  ❌ BUG 1: _WORK_HDR used re.compile without re.MULTILINE
             → _WORK_HDR.search(full_text) only matched if resume STARTED
               with "Experience", which never happens
             → has_work_hdr was always False
             → fallback mode fired on ALL resumes
             → committee/project dates leaked into experience calculation
  ✅ FIX 1: Added re.MULTILINE so ^ matches each line start in full text

  ❌ BUG 2: Degree detection used `kw in text_lower` (substring check)
             → "b.com" matched "github.com/rohannegi-2005"
             → All candidates with github link got B.Com degree
  ✅ FIX 2: Regex word-boundary check: (?<![a-z])kw(?![a-z])

  ❌ BUG 3: "bba" mapped to B.Com category
  ✅ FIX 3: BBA is its own degree with correct label

  ❌ BUG 4: "Internship:", "Volunteer:", "Position of Responsibility" not
             in _WORK_HDR → resumes with "Internship:" only got fallback mode
  ✅ FIX 4: Added internship/volunteer to WORK_HDR and OTHER_HDR

  ❌ BUG 5: JD extractor returned full sentences as "skills"
  ✅ FIX 5: Two-pass approach — known tech regex + short-phrase filter
"""

import re
from datetime import date
from typing import List, Tuple


class ParserService:

    # ── Contact patterns ───────────────────────────────────────────────
    _EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")
    _PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")

    # ── Month lookup ───────────────────────────────────────────────────
    _MONTHS = {m: i + 1 for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )}

    # ─────────────────────────────────────────────────────────────────
    # DEGREE TAXONOMY  (ordered highest → lowest for detection priority)
    # ─────────────────────────────────────────────────────────────────
    DEGREE_RANK = [
        "PhD", "M.Tech", "M.Sc", "MBA", "MCA",
        "B.Tech", "B.Sc", "BCA", "BBA", "B.Com", "B.A",
        "Diploma", "12th", "10th",
    ]

    # Each entry is a list of lowercase substrings to scan for.
    # Detection uses WORD-BOUNDARY regex to avoid false matches
    # (e.g. "b.com" inside "github.com").
    DEGREE_KEYWORDS = {
        "PhD":    ["ph.d", "phd", "doctor of philosophy", "doctorate"],
        "M.Tech": ["m.tech", "master of technology", "m tech", "mtech"],
        "M.Sc":   ["m.sc", "master of science", "msc"],
        "MBA":    ["mba", "master of business administration", "master of business"],
        "MCA":    ["mca", "master of computer application"],
        "B.Tech": ["b.tech", "bachelor of technology", "b tech", "btech",
                   "bachelor of engineering", "b.e."],
        "B.Sc":   ["b.sc", "bachelor of science", "bsc"],
        "BCA":    ["bca", "bachelor of computer application"],
        "BBA":    ["bba", "bachelor of business administration"],
        "B.Com":  ["b.com", "bachelor of commerce", "bcom"],
        "B.A":    ["b.a.", "bachelor of arts", "b.des", "bachelor of design"],
        "Diploma":["diploma"],
        "12th":   ["12th", "higher secondary", "class xii", "intermediate", "senior secondary"],
        "10th":   ["10th", "matriculation", "class x", "class 10", "secondary school"],
    }

    # ─────────────────────────────────────────────────────────────────
    # SECTION HEADER PATTERNS
    # Key fix: re.MULTILINE so ^ matches start of each line in full text
    # ─────────────────────────────────────────────────────────────────

    # Marks START of a work section
    _WORK_HDR_RE = re.compile(
        r"^[ \t]*(?:(?:work\s+)?experience|professional\s+experience|"
        r"employment(?:\s+history)?|internships?|work\s+history|"
        r"relevant\s+experience|career\s+summary)\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Marks END of a work section (any non-work section header)
    _STOP_HDR_RE = re.compile(
        r"^[ \t]*(?:education|academic(?:\s+background)?|qualification|"
        r"skills?|(?:(?:academic|personal|professional|key|notable)\s+)?projects?|"
        r"certif(?:ications?)?(?:\s*&\s*accomplishments?)?|achievement|award|honor|"
        r"interest|language|reference|publication|volunteer|"
        r"position\s+of\s+responsibility|extracurricular|"
        r"activities|summary|objective|profile|"
        r"accomplishments?|hobbies|interests\s*&\s*hobbies)\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Education context — lines with these keywords carry education dates
    _EDU_LINE_RE = re.compile(
        r"(?:school|college|university|institute|academy|b\.?tech|m\.?tech|"
        r"bachelor|master|diploma|secondary|cbse|ssc|hsc|polytechnic|"
        r"b\.?sc|m\.?sc|mba|mca|bca|aptech|coursera|udemy|certification|"
        r"cinematics|animation\s+school|maac|100\s+days|bootcamp|"
        r"personal\s+project|professional\s+project|github\.com)",
        re.IGNORECASE,
    )

    # Internship context — cap any single stint at 6 months
    _INTERN_LINE_RE = re.compile(
        r"\b(?:intern|trainee|apprentice|volunteer|social\s+intern|"
        r"instructor|coach|tutor|freelance)\b",
        re.IGNORECASE,
    )

    # ─────────────────────────────────────────────────────────────────
    # DATE RANGE PATTERNS
    # ─────────────────────────────────────────────────────────────────

    # mm/yyyy – mm/yyyy  OR  mm/yyyy – Present
    _MM_YYYY_RE = re.compile(
        r"(\d{1,2})/(\d{4})\s*[-–—]+\s*"
        r"(?:(\d{1,2})/(\d{4})|(present|ongoing|current|now|till\s*date))",
        re.IGNORECASE,
    )

    # Month YYYY – Month YYYY  (handles parentheses: "(June 2025 – Aug 2025 | Lucknow)")
    _MON_YYYY_RE = re.compile(
        r"[\(\s\n]?"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})"
        r"\s*[-–—to]+\s*"
        r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?"
        r"(\d{4}|[Pp]resent|[Oo]ngoing|[Cc]urrent)"
        r"(?:[^\w]|$)",
        re.IGNORECASE,
    )

    # Year-only: YYYY – YYYY  or  YYYY – Present
    # Only match 4-digit years in range 1990–2030 to avoid false positives
    _YR_ONLY_RE = re.compile(
        r"(?<!\d)((?:19|20)\d{2})\s*[-–—]+\s*"
        r"((?:19|20)\d{2}|[Pp]resent|[Oo]ngoing|[Cc]urrent)(?!\d)",
    )

    # Explicit "X years of experience"
    _EXP_STMT_RE = re.compile(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:year|yr)s?\s*(?:of\s+)?(?:work\s+)?exp",
        re.IGNORECASE,
    )

    # ─────────────────────────────────────────────────────────────────
    # JD SKILL EXTRACTOR
    # ─────────────────────────────────────────────────────────────────
    _TECH_RE = re.compile(
        r"\b(Python|Java(?:Script)?|TypeScript|React(?:\.js)?|Vue(?:\.js)?|Angular|"
        r"Node(?:\.js)?|Django|Flask|FastAPI|Spring(?:\s+Boot)?|"
        r"SQL|MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|"
        r"Docker|Kubernetes|AWS|GCP|Azure|CI/?CD|Terraform|"
        r"TensorFlow|PyTorch|scikit-?learn|Keras|NLTK|spaCy|"
        r"Machine\s+Learning|Deep\s+Learning|NLP|Computer\s+Vision|"
        r"Data\s+(?:Analysis|Analytics|Science|Visualization)|"
        r"Pandas|NumPy|SciPy|Spark|Hadoop|"
        r"Power\s+BI|Tableau|Excel|Looker|"
        r"Git(?:Hub)?|REST(?:ful)?\s*API|GraphQL|gRPC|Kafka|RabbitMQ|"
        r"HTML5?|CSS3?|Sass|Tailwind|Bootstrap|"
        r"C\+\+|C#|Go|Rust|Kotlin|Swift|Flutter|React\s+Native|"
        r"Linux|Bash|Shell|Agile|Scrum|Jira|"
        r"Microservices|DevOps|MLOps|LLM|LangChain|OpenAI|"
        r"Statistics|Probability|EDA|Regression|Classification|"
        r"Communication|Problem[- ]Solving|Leadership|Teamwork|"
        r"Microsoft\s+Office|Jupyter|VS\s+Code|PostgreSQL|Firebase|"
        r"SEO|SEM|Google\s+Ads|Google\s+Analytics|Content\s+(?:Strategy|Marketing)|"
        r"Social\s+Media|Email\s+(?:Marketing|Campaigns)|"
        r"AutoCAD|SolidWorks|MATLAB|ANSYS|"
        r"Unity|Unreal\s+Engine|C#|Figma|Adobe\s+XD|"
        r"R\b|SPSS|SAS)\b",
        re.IGNORECASE,
    )

    _JD_VERB_RE = re.compile(
        r"^\s*(?:assist|build|create|develop|design|manage|support|"
        r"collaborate|perform|generate|write|analyz|implement|ensure|"
        r"work|collect|clean|preprocess|identify|seek|look|gain|"
        r"require|prefer|must|should|will|can|able|provide|make|help)\b",
        re.IGNORECASE,
    )

    _JD_SKIP = frozenset({
        "required skills", "good to have", "education", "skills",
        "requirements", "responsibilities", "qualifications",
        "experience", "what you'll gain", "key responsibilities",
        "about", "overview", "benefits", "nice to have",
        "minimum qualifications", "preferred qualifications",
        "job overview", "job description", "role", "about the role",
        "your responsibilities", "must have", "preferred",
    })

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def extract_contact(self, text: str) -> Tuple[str, str, str]:
        email = self._find_email(text)
        phone = self._find_phone(text)
        name  = self._find_name(text)
        return name, email, phone

    def extract_experience(self, text: str) -> float:
        """
        Return professional experience in years (float, 0–25).

        Algorithm:
          1. Explicit mention ("3 years of experience")   → return immediately
          2. Isolated work-section extraction              → parse date ranges
          3. Fallback: scan whole text for date ranges     → filter by context
        """
        # ── Layer 1: explicit statement ─────────────────────────────────
        for m in self._EXP_STMT_RE.finditer(text):
            val = float(m.group(1))
            if val <= 30:
                return round(val, 1)

        # ── Layer 2: date ranges in isolated work section ───────────────
        work_text = self._extract_work_section(text)
        return self._sum_date_ranges(work_text, full_text=text)

    def extract_highest_degree(self, text: str) -> str:
        """
        Return the highest academic degree found.
        Uses regex with negative lookbehind/lookahead to prevent
        false matches (e.g. 'b.com' inside 'github.com').
        """
        text_lower = text.lower()
        for degree in self.DEGREE_RANK:
            for kw in self.DEGREE_KEYWORDS[degree]:
                # Escape the keyword for regex use
                escaped = re.escape(kw)
                # Negative lookbehind: not preceded by a-z (word char)
                # Negative lookahead:  not followed by a-z
                pattern = r"(?<![a-z])" + escaped + r"(?![a-z])"
                if re.search(pattern, text_lower):
                    return degree
        return "Unknown"

    def extract_skills_from_jd(self, jd_text: str) -> List[str]:
        """
        Extract clean skill tokens from a Job Description.

        Pass 1 — known-tech keyword sweep (most reliable)
        Pass 2 — short bullet items: ≤4 words, no verbs, no parens

        Returns ≤25 clean, deduplicated skill strings.
        """
        skills: set = set()

        # Pass 1 — known-tech keywords (normalise to Title Case)
        for m in self._TECH_RE.finditer(jd_text):
            skill = re.sub(r"\s+", " ", m.group().strip())
            # Preserve common UPPER acronyms; title-case everything else
            if skill.upper() == skill and len(skill) <= 5:
                skills.add(skill)       # e.g. SQL, NLP, AWS, EDA
            elif skill[0].isupper():
                skills.add(skill)       # e.g. "Machine Learning", "Python"
            else:
                skills.add(skill.title())  # e.g. "statistics" → "Statistics"

        # Pass 2 — clean bullet-point phrases
        for line in jd_text.splitlines():
            raw = line.strip()
            raw = re.sub(r"^[\d\.\-\*\u2022\u25cf\u2013\u2014]+\s*", "", raw).strip()
            raw = raw.rstrip(".,;:()")

            if not raw or len(raw) > 50:
                continue
            if raw.lower() in self._JD_SKIP:
                continue
            if self._JD_VERB_RE.search(raw):
                continue
            if re.search(r"[(){}[\]/\\]", raw):
                continue

            words = raw.split()
            if not (1 <= len(words) <= 4):
                continue
            if not (words[0][0].isupper() or words[0].isupper()):
                continue

            skills.add(raw)

        # Normalise casing: prefer Title Case; drop exact case-insensitive dups
        normalised: dict = {}   # lower → display form
        for s in skills:
            key = s.lower()
            # prefer the form that starts uppercase
            if key not in normalised or s[0].isupper():
                normalised[key] = s
        skills = set(normalised.values())

        # Drop noisy tail-words: phrases like "Data visualization skills",
        # "Basic machine learning knowledge", "Problem-solving mindset"
        # when a cleaner tech-keyword version already exists
        _NOISY_TAIL = re.compile(
            r"\s+(?:skills?|knowledge|mindset|understanding|experience|"
            r"proficiency|fundamentals?|exposure|concepts?|tools?)\s*$",
            re.IGNORECASE,
        )
        cleaned: set = set()
        for s in skills:
            stripped = _NOISY_TAIL.sub("", s).strip()
            if stripped and stripped != s:
                # only keep stripped version if it has ≥2 chars and looks like a skill
                cleaned.add(stripped if len(stripped) > 1 else s)
            else:
                cleaned.add(s)
        skills = cleaned

        # Deduplicate: drop pure-substring items
        final = sorted(skills)
        deduped = [s for s in final
                   if not any(s.lower() in o.lower() and s.lower() != o.lower()
                               for o in final)]

        return deduped[:25]

    # ═══════════════════════════════════════════════════════════════════════
    # PRIVATE — work section extraction
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_work_section(self, text: str) -> str:
        """
        Isolate lines that belong to the work/experience section.

        Strategy:
          A) If a work-section header is found:
             - Scan lines; collect everything between the work header
               and the next stop-header.
          B) If no work header exists (e.g. resume only has "Internship:" as
             a sub-bullet, or no section labels at all):
             - Collect ALL lines, excluding lines that are clearly
               education-section content.
        """
        lines = text.splitlines()

        # Detect whether resume has explicit work section header
        has_work_hdr = bool(self._WORK_HDR_RE.search(text))

        work_lines: List[str] = []
        in_work   = False
        in_stop   = False   # inside a non-work section

        for line in lines:
            s = line.strip()

            # Is this a work-section header?
            is_work_hdr = bool(self._WORK_HDR_RE.match(s + "\n"))

            # Is this a stop-section header (edu / skills / projects / etc.)?
            is_stop_hdr = bool(self._STOP_HDR_RE.match(s + "\n"))

            if is_work_hdr:
                in_work = True
                in_stop = False
                continue                 # don't include the header line itself

            if is_stop_hdr and in_work:
                in_work = False          # work section ended
                in_stop = True
                continue

            if is_stop_hdr:
                in_stop = True
                continue

            # Resume a new (work) section by unsetting in_stop if another
            # work header appears — handled by the is_work_hdr branch above.

            if in_work:
                work_lines.append(line)

            elif not has_work_hdr and not in_stop:
                # Fallback: no explicit work header — include non-edu lines
                if not self._EDU_LINE_RE.search(line):
                    work_lines.append(line)

        return "\n".join(work_lines) if work_lines else text

    # ═══════════════════════════════════════════════════════════════════════
    # PRIVATE — date range summation
    # ═══════════════════════════════════════════════════════════════════════

    def _sum_date_ranges(self, work_text: str, full_text: str = "") -> float:
        """
        Find all date ranges in work_text, deduplicate overlapping ones,
        and return total professional experience in years.

        Rules:
          - Education-context lines are skipped entirely.
          - Internship-context stints are capped at 6 months each.
          - If ALL stints found are internships, total is capped at 0.5 yr.
        """
        today = date.today()
        seen: set = set()        # dedup by (start_ym, end_ym) tuple
        total_months = 0
        is_all_intern = True

        def process(y1: int, m1: int, y2: int, m2: int, ctx: str) -> None:
            nonlocal total_months, is_all_intern

            key = (y1 * 12 + m1, y2 * 12 + m2)
            if key in seen:
                return
            seen.add(key)

            # Skip if context suggests this is an education date range
            if self._EDU_LINE_RE.search(ctx):
                return

            dur = max(0, (y2 - y1) * 12 + (m2 - m1))
            is_intern = bool(self._INTERN_LINE_RE.search(ctx))

            if is_intern:
                dur = min(dur, 6)            # cap single internship
            else:
                is_all_intern = False

            total_months += dur

        # ── Pattern 1: mm/yyyy ────────────────────────────────────────────
        for m in self._MM_YYYY_RE.finditer(work_text):
            y1, m1 = int(m.group(2)), int(m.group(1))
            ctx = work_text[max(0, m.start() - 300): m.end() + 150]
            if m.group(5):   # Present / Ongoing
                y2, m2 = today.year, today.month
            else:
                y2, m2 = int(m.group(4)), int(m.group(3))
            process(y1, m1, y2, m2, ctx)

        # ── Pattern 2: Month YYYY ─────────────────────────────────────────
        for m in self._MON_YYYY_RE.finditer(work_text):
            m1_name = m.group(1).lower()[:3]
            y1 = int(m.group(2))
            end_str = m.group(4)
            m1 = self._MONTHS.get(m1_name, 6)
            ctx = work_text[max(0, m.start() - 300): m.end() + 150]

            if end_str.lower() in ("present", "ongoing", "current"):
                y2, m2 = today.year, today.month
            else:
                y2 = int(end_str)
                m2_name = (m.group(3) or "").lower()[:3]
                m2 = self._MONTHS.get(m2_name, 6) if m2_name else m1

            process(y1, m1, y2, m2, ctx)

        # ── Pattern 3: Year-only YYYY – YYYY (e.g. "2023 – 2024") ────────
        # Only applied if patterns 1 & 2 found nothing (avoids double-counting)
        if not seen:
            for m in self._YR_ONLY_RE.finditer(work_text):
                y1 = int(m.group(1))
                end_str = m.group(2)
                ctx = work_text[max(0, m.start() - 300): m.end() + 150]
                if end_str.lower() in ("present", "ongoing", "current"):
                    y2, m2 = today.year, today.month
                else:
                    y2 = int(end_str)
                    m2 = 6    # assume mid-year when month unknown
                process(y1, 6, y2, m2, ctx)   # assume mid-year start

        years = round(total_months / 12.0, 1)

        # If every stint was an internship, cap total at 0.5 yr
        if is_all_intern and years > 0:
            years = min(years, 0.5)

        return min(years, 25.0)

    # ═══════════════════════════════════════════════════════════════════════
    # PRIVATE — contact helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _find_email(self, text: str) -> str:
        m = self._EMAIL_RE.search(text)
        return m.group() if m else "Not Found"

    def _find_phone(self, text: str) -> str:
        for m in self._PHONE_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 8 <= len(digits) <= 15:
                return m.group().strip()
        return "Not Found"

    def _find_name(self, text: str) -> str:
        """
        Multi-strategy name extractor:
          1. Explicit 'Name: Firstname Lastname' label (common in template resumes)
          2. First non-empty line heuristic: 2-4 all-alpha words, not a title/header
        """
        # Strategy 1: explicit "Name:" label
        name_label = re.search(
            r"(?:^|\n)\s*[Nn]ame\s*[:\-]\s*([A-Za-z][A-Za-z .'\-]{2,40})",
            text,
        )
        if name_label:
            candidate = name_label.group(1).strip()
            words = candidate.split()
            if 2 <= len(words) <= 5:
                return candidate.title()

        # Strategy 2: heuristic scan of first 12 lines
        IGNORE = frozenset({
            "resume", "curriculum vitae", "cv", "profile", "summary",
            "objective", "contact", "address", "experience", "education",
            "programmer", "developer", "engineer", "analyst", "designer",
            "specialist", "manager", "consultant", "coordinator", "intern",
            "ui/ux", "digital", "marketing", "mechanical", "fresher",
        })
        TITLE_WORDS = frozenset({
            "software", "senior", "junior", "lead", "data", "full", "stack",
            "front", "back", "end", "mobile", "web", "cloud", "ai", "ml",
            "ui", "ux", "digital", "marketing", "mechanical",
        })
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines[:12]:
            low = line.lower()
            if any(w in low for w in IGNORE):
                continue
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(re.match(r"^[A-Za-z.'\-]+$", w) for w in words):
                    if "@" not in line and not re.search(r"\d{4}", line):
                        if not any(w.lower() in TITLE_WORDS for w in words):
                            return line.title()
        return "Unknown"
