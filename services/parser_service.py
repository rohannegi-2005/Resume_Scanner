"""
ParserService v4 — Template-aware SaaS-grade parser

Template family detected (UPES/similar format):
  Line 0:  Full Name
  Line 1:  email | phone | linkedin | github
  Sections: Professional Summary → Education → Technical Skills →
            Experience/Internships → Projects →
            Certifications & Achievements → Positions of Responsibility
  Experience format: Company DateRange\n  Role: X | Duration: Y Years [Z Months]

v4 fixes over v3:
  ❌ BUG 1: Duration field ("Duration: 2 Years") completely ignored
            → entire template family's explicit exp field thrown away
  ✅ FIX 1: _DURATION_RE parses "Duration: X Years [Y Months]" as Priority 1
            date ranges become fallback only

  ❌ BUG 2: "Certifications & Achievements" not a stop header
            → cert section dates (GDC 2024, EthIndia 2023) leaked into work
  ✅ FIX 2: pattern now covers both "& Achievements" and "& Accomplishments"

  ❌ BUG 3: "Positions of Responsibility" absent from stop headers
  ✅ FIX 3: added to _STOP_HDR_FULL_RE

  ❌ BUG 4: CGPA not in edu-context filter
            → "IIT Kharagpur | CGPA: 8.78/10 Jul 2016–May 2020" not filtered
  ✅ FIX 4: added "cgpa" and "| cgpa" to _EDU_LINE_RE

  ❌ BUG 5: Akarsh 0 experience — single-date internship with "Duration: 8 Weeks"
            has no date range; neither pattern matched
  ✅ FIX 5: Duration parser handles Weeks-only → capped at 0.5 yr total

  ❌ BUG 6: Global intern cap — 2 × 6-month internships = 0.5 yr (wrong)
  ✅ FIX 6: Per-stint 6-month cap; only cap TOTAL when NO professional jobs found
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

    DEGREE_KEYWORDS = {
        "PhD":     ["ph.d", "phd", "doctor of philosophy", "doctorate"],
        "M.Tech":  ["m.tech", "master of technology", "m tech", "mtech",
                    "m.s. in", "m.s in", "master of science in"],
        "M.Sc":    ["m.sc", "master of science", "msc"],
        "MBA":     ["mba", "master of business administration", "master of business"],
        "MCA":     ["mca", "master of computer application"],
        "B.Tech":  ["b.tech", "bachelor of technology", "b tech", "btech",
                    "bachelor of engineering", "b.e.",
                    "b.tech in"],
        "B.Sc":    ["b.sc", "bachelor of science", "bsc", "b.sc."],
        "BCA":     ["bca", "bachelor of computer application"],
        "BBA":     ["bba", "bachelor of business administration"],
        "B.Com":   ["b.com", "bachelor of commerce", "bcom"],
        "B.A":     ["b.a.", "bachelor of arts", "b.des", "bachelor of design"],
        "Diploma": ["diploma"],
        "12th":    ["12th", "higher secondary", "class xii", "intermediate",
                    "senior secondary", "hsc"],
        "10th":    ["10th", "matriculation", "class x", "class 10",
                    "secondary school", "ssc"],
    }

    # ─────────────────────────────────────────────────────────────────
    # SECTION HEADER PATTERNS
    # ─────────────────────────────────────────────────────────────────

    _WORK_HDR_RE = re.compile(
        r"^[ \t]*(?:(?:work\s+)?experience|professional\s+experience|"
        r"employment(?:\s+history)?|internships?|work\s+history|"
        r"relevant\s+experience|career\s+summary|work)\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Full-line stop headers — must match the ENTIRE trimmed line
    _STOP_HDR_FULL_RE = re.compile(
        r"^[ \t]*(?:"
        r"education|academic(?:\s+background)?|qualification|"
        r"skills?|key\s+skills?|technical\s+skills?|"
        r"(?:(?:academic|personal|professional|key|notable)\s+)?projects?|"
        # certifications: handles "& Achievements" AND "& Accomplishments"
        r"certif(?:ications?(?:\s+and\s+accomplishments?)?)?(?:\s*&\s*(?:achievements?|accomplishments?))?|"
        r"achievement|award|honor|interest|language|reference|publication|volunteer|"
        # positions of responsibility (template family)
        r"positions?\s+of\s+responsibility|"
        r"extracurricular|activities|summary|objective|profile|"
        r"accomplishments?|hobbies|interests\s*&\s*hobbies"
        r")\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Prefix-only stop — catches merged two-column headers
    # e.g. "EDUCATION Master of Science Jun 2019" or "KEY SKILLS Selenium..."
    _STOP_HDR_PREFIX_RE = re.compile(
        r"^[ \t]*(?:education|key\s+skills?|technical\s+skills?|skills?)\b",
        re.IGNORECASE,
    )

    # Education-context lines — dates on these lines are academic, not work
    _EDU_LINE_RE = re.compile(
        r"(?:cgpa|c\.g\.p\.a|gpa|school|college|university|institute|academy|"
        r"b\.?tech|m\.?tech|bachelor|master|diploma|secondary|cbse|ssc|hsc|"
        r"polytechnic|b\.?sc|m\.?sc|mba|mca|bca|aptech|coursera|udemy|"
        r"certification|cinematics|animation\s+school|maac|100\s+days|bootcamp|"
        r"personal\s+project|professional\s+project|github\.com)",
        re.IGNORECASE,
    )

    # Internship-context lines — cap each at 6 months
    _INTERN_LINE_RE = re.compile(
        r"\b(?:intern|trainee|apprentice|volunteer|social\s+intern|"
        r"instructor|coach|tutor|freelance)\b",
        re.IGNORECASE,
    )

    # ─────────────────────────────────────────────────────────────────
    # DURATION FIELD PARSER  (v4 — Priority 1 for template family)
    #
    # Matches:
    #   "Duration: 2 Years"
    #   "Duration: 1 Year 9 Months"
    #   "Duration: 6 Months"
    #   "Duration: 8 Weeks"
    # ─────────────────────────────────────────────────────────────────
    _DURATION_RE = re.compile(
        r"Duration:\s*"
        r"(?:"
        r"(?P<years>\d+)\s+Years?(?:\s+(?P<months_a>\d+)\s+Months?)?"   # Y yrs [M months]
        r"|(?P<months_b>\d+)\s+Months?"                                   # M months only
        r"|(?P<weeks>\d+)\s+Weeks?"                                       # N weeks only
        r")",
        re.IGNORECASE,
    )

    # ─────────────────────────────────────────────────────────────────
    # DATE RANGE PATTERNS  (fallback when no Duration fields present)
    # ─────────────────────────────────────────────────────────────────

    # mm/yyyy – mm/yyyy  OR  mm/yyyy – Present
    _MM_YYYY_RE = re.compile(
        r"(\d{1,2})/(\d{4})\s*[-–—]+\s*"
        r"(?:(\d{1,2})/(\d{4})|(present|ongoing|current|now|till\s*date))",
        re.IGNORECASE,
    )

    # Month YYYY – Month YYYY  or  Month YYYY – Present
    _MON_YYYY_RE = re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})"
        r"\s*[-–—to]+\s*"
        r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?"
        r"(\d{4}|[Pp]resent|[Oo]ngoing|[Cc]urrent)"
        r"(?:[^\w]|$)",
        re.IGNORECASE,
    )

    # Year-only YYYY – YYYY (last-resort fallback)
    _YR_ONLY_RE = re.compile(
        r"(?<!\d)((?:19|20)\d{2})\s*[-–—]+\s*"
        r"((?:19|20)\d{2}|[Pp]resent|[Oo]ngoing|[Cc]urrent)(?!\d)",
    )

    # Explicit "X years of experience" — checked before everything else
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
        r"SEO|SEM|Google\s+Ads|Google\s+Analytics|"
        r"Content\s+(?:Strategy|Marketing)|Social\s+Media|"
        r"Email\s+(?:Marketing|Campaigns)|"
        r"AutoCAD|SolidWorks|MATLAB|ANSYS|"
        r"Unity|Unreal\s+Engine|Figma|Adobe\s+XD|"
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

        Priority order:
          1. Explicit "X years of experience" in text (immediate return)
          2. Duration fields in work section  ← KEY FIX for template family
          3. Date range parsing               ← fallback for other formats
        """
        # ── Priority 1: explicit statement (e.g. "4 years of experience") ─
        for m in self._EXP_STMT_RE.finditer(text):
            val = float(m.group(1))
            if 0 < val <= 30:
                return round(val, 1)

        work_text = self._extract_work_section(text)

        # ── Priority 2: Duration fields (template family explicit signal) ──
        duration_years = self._sum_duration_fields(work_text)
        if duration_years > 0:
            return duration_years

        # ── Priority 3: date range parsing (all other formats) ─────────────
        return self._sum_date_ranges(work_text)

    def extract_highest_degree(self, text: str) -> str:
        text_lower = text.lower()
        for degree in self.DEGREE_RANK:
            for kw in self.DEGREE_KEYWORDS[degree]:
                escaped = re.escape(kw)
                pattern = r"(?<![a-z])" + escaped + r"(?![a-z])"
                if re.search(pattern, text_lower):
                    return degree
        return "Unknown"

    def extract_skills_from_jd(self, jd_text: str) -> List[str]:
        """Extract clean skill tokens from a Job Description (≤25 results)."""
        skills: set = set()

        # Pass 1 — known-tech keyword sweep
        for m in self._TECH_RE.finditer(jd_text):
            skill = re.sub(r"\s+", " ", m.group().strip())
            if skill.upper() == skill and len(skill) <= 5:
                skills.add(skill)
            elif skill[0].isupper():
                skills.add(skill)
            else:
                skills.add(skill.title())

        # Pass 2 — short bullet items (≤4 words, no verbs, no parens)
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

        # Normalise casing
        normalised: dict = {}
        for s in skills:
            key = s.lower()
            if key not in normalised or s[0].isupper():
                normalised[key] = s
        skills = set(normalised.values())

        # Strip noisy suffix words
        _NOISY = re.compile(
            r"\s+(?:skills?|knowledge|mindset|understanding|experience|"
            r"proficiency|fundamentals?|exposure|concepts?|tools?)\s*$",
            re.IGNORECASE,
        )
        cleaned: set = set()
        for s in skills:
            stripped = _NOISY.sub("", s).strip()
            cleaned.add(stripped if stripped and len(stripped) > 1 else s)
        skills = cleaned

        final = sorted(skills)
        deduped = [s for s in final
                   if not any(s.lower() in o.lower() and s.lower() != o.lower()
                               for o in final)]
        return deduped[:25]

    # ═══════════════════════════════════════════════════════════════════════
    # PRIVATE — Duration-field parser  (v4 addition)
    # ═══════════════════════════════════════════════════════════════════════

    def _sum_duration_fields(self, work_text: str) -> float:
        """
        Parse all "Duration: ..." fields found in the work section and return
        total professional experience in years.

        Rules:
          • "Duration: 2 Years"           → 24 months
          • "Duration: 1 Year 9 Months"   → 21 months
          • "Duration: 6 Months"          → 6 months
          • "Duration: 8 Weeks"           → treated as short internship (1 month)
            If ONLY weeks-based durations are found, cap total at 0.5 yr.
        """
        total_months = 0
        weeks_only   = True       # stays True until we see a year/month duration

        for m in self._DURATION_RE.finditer(work_text):
            yrs  = int(m.group("years")    or 0)
            mths = int(m.group("months_a") or m.group("months_b") or 0)
            wks  = int(m.group("weeks")    or 0)

            if wks and not yrs and not mths:
                # weeks-only entry — count as 1 month max
                total_months += 1
            else:
                weeks_only    = False
                total_months += yrs * 12 + mths

        if total_months == 0:
            return 0.0

        years = round(total_months / 12.0, 1)

        # If we only found week-based internships, cap at 0.5 yr
        if weeks_only:
            years = min(years, 0.5)

        return min(years, 25.0)

    # ═══════════════════════════════════════════════════════════════════════
    # PRIVATE — work section extraction
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_work_section(self, text: str) -> str:
        """
        Isolate lines belonging to the work/experience section using a
        line-level state machine.

        Handles:
          A) Clean single-column resumes with explicit section headers
          B) Two-column PDFs where "WORK" and "EXPERIENCE" are on separate lines
          C) Merged-header lines like "EDUCATION Master of Science Jun 2019"
        """
        lines = text.splitlines()
        has_work_hdr = bool(self._WORK_HDR_RE.search(text))

        work_lines: List[str] = []
        in_work = False
        in_stop = False

        for line in lines:
            s = line.strip()

            is_work_hdr = bool(self._WORK_HDR_RE.match(s + "\n"))
            is_stop_hdr = (
                bool(self._STOP_HDR_FULL_RE.match(s + "\n"))
                or bool(self._STOP_HDR_PREFIX_RE.match(s))
            )

            if is_work_hdr:
                in_work = True
                in_stop = False
                continue

            if is_stop_hdr and in_work:
                in_work = False
                in_stop = True
                continue

            if is_stop_hdr:
                in_stop = True
                continue

            if in_work:
                work_lines.append(line)
            elif not has_work_hdr and not in_stop:
                # No explicit work header — include non-education content
                if not self._EDU_LINE_RE.search(line):
                    work_lines.append(line)

        return "\n".join(work_lines) if work_lines else text

    # ═══════════════════════════════════════════════════════════════════════
    # PRIVATE — date range summation  (fallback for non-template resumes)
    # ═══════════════════════════════════════════════════════════════════════

    def _sum_date_ranges(self, work_text: str) -> float:
        """
        Parse date ranges from work_text and return total years.

        Per-stint rules:
          - Education-context lines → skipped
          - Internship-context stints → capped at 6 months each
          - If NO professional stints found at all → cap total at 0.5 yr
        """
        today = date.today()
        seen:  set = set()
        total_months  = 0
        has_prof_work = False   # True when ≥1 non-intern stint found

        def process(y1: int, m1: int, y2: int, m2: int, ctx: str) -> None:
            nonlocal total_months, has_prof_work
            key = (y1 * 12 + m1, y2 * 12 + m2)
            if key in seen:
                return
            seen.add(key)
            if self._EDU_LINE_RE.search(ctx):
                return
            dur = max(0, (y2 - y1) * 12 + (m2 - m1))
            if bool(self._INTERN_LINE_RE.search(ctx)):
                dur = min(dur, 6)          # per-stint internship cap
            else:
                has_prof_work = True
            total_months += dur

        # Pattern 1: mm/yyyy
        for m in self._MM_YYYY_RE.finditer(work_text):
            y1, m1 = int(m.group(2)), int(m.group(1))
            ctx = work_text[max(0, m.start() - 300): m.end() + 150]
            y2, m2 = (today.year, today.month) if m.group(5) \
                     else (int(m.group(4)), int(m.group(3)))
            process(y1, m1, y2, m2, ctx)

        # Pattern 2: Month YYYY
        for m in self._MON_YYYY_RE.finditer(work_text):
            m1n = m.group(1).lower()[:3]
            y1  = int(m.group(2))
            end = m.group(4)
            m1  = self._MONTHS.get(m1n, 6)
            ctx = work_text[max(0, m.start() - 300): m.end() + 150]
            if end.lower() in ("present", "ongoing", "current"):
                y2, m2 = today.year, today.month
            else:
                y2   = int(end)
                m2n  = (m.group(3) or "").lower()[:3]
                m2   = self._MONTHS.get(m2n, 6) if m2n else m1
            process(y1, m1, y2, m2, ctx)

        # Pattern 3: Year-only YYYY–YYYY (last resort, only if nothing found yet)
        if not seen:
            for m in self._YR_ONLY_RE.finditer(work_text):
                y1  = int(m.group(1))
                end = m.group(2)
                ctx = work_text[max(0, m.start() - 300): m.end() + 150]
                if end.lower() in ("present", "ongoing", "current"):
                    y2, m2 = today.year, today.month
                else:
                    y2, m2 = int(end), 6
                process(y1, 6, y2, m2, ctx)

        years = round(total_months / 12.0, 1)

        # Cap at 0.5 yr only when EVERY detected stint was an internship
        if not has_prof_work and years > 0:
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
        Strategy 1: Explicit 'Name: Firstname Lastname' label.
        Strategy 2: Template family — Line 0 is always the name if Line 1
                    contains a pipe-separated contact string with email.
        Strategy 3: Heuristic scan of first 12 non-empty lines.
        """
        # Strategy 1: explicit label
        m = re.search(
            r"(?:^|\n)\s*[Nn]ame\s*[:\-]\s*([A-Za-z][A-Za-z .'\\-]{2,40})",
            text,
        )
        if m:
            candidate = m.group(1).strip()
            words = candidate.split()
            if 2 <= len(words) <= 5:
                return candidate.title()

        # Strategy 2: template family fingerprint
        # Line 0 = name, Line 1 = "x@y.com | +91..." with pipe separator
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 2:
            line0, line1 = lines[0], lines[1]
            has_pipe    = "|" in line1
            has_email   = bool(self._EMAIL_RE.search(line1))
            words0      = line0.split()
            is_name_like = (
                2 <= len(words0) <= 4
                and all(re.match(r"^[A-Za-z.'\\-]+$", w) for w in words0)
                and "@" not in line0
            )
            if has_pipe and has_email and is_name_like:
                return line0.title()

        # Strategy 3: heuristic scan
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
            "qa", "automation", "quality",
        })
        for line in lines[:12]:
            low = line.lower()
            if any(w in low for w in IGNORE):
                continue
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(re.match(r"^[A-Za-z.'\\-]+$", w) for w in words):
                    if "@" not in line and not re.search(r"\d{4}", line):
                        if not any(w.lower() in TITLE_WORDS for w in words):
                            return line.title()
        return "Unknown"
