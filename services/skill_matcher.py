"""
SkillMatcher: 3-layer skill matching pipeline
  Layer 1 — Exact / substring match (instant, confidence = 1.0)
  Layer 2 — Fuzzy match via FuzzyWuzzy  (fast, threshold 82)
  Layer 3 — Semantic match via Sentence Transformers (accurate, threshold 0.52)

Tuned for all-MiniLM-L6-v2. The semantic threshold of 0.52 catches synonyms
like "ML" ↔ "machine learning", "NLP" ↔ "natural language processing", etc.
"""
from typing import List, Tuple
from fuzzywuzzy import fuzz


class SkillMatcher:
    FUZZY_THRESHOLD = 82      # % — lower = more false positives
    SEMANTIC_THRESHOLD = 0.52  # cosine similarity — tuned for MiniLM

    def __init__(self, ml_engine):
        self.ml = ml_engine

    def match_skills(
        self,
        resume_text: str,
        required_skills: List[str],
    ) -> Tuple[List[Tuple[str, float]], List[str]]:
        """
        Returns:
            matched   — list of (skill_name, confidence_0_to_1)
            unmatched — list of skill names that weren't found
        """
        matched: List[Tuple[str, float]] = []
        unmatched: List[str] = []
        resume_lower = resume_text.lower()

        # Pre-encode resume sentences once (avoid re-encoding per skill)
        sentences = self._extract_sentences(resume_text)
        sent_embeddings = None
        if sentences:
            try:
                sent_embeddings = self.ml.encode(sentences)
            except Exception:
                pass

        for skill in required_skills:
            skill_lower = skill.lower().strip()

            # --- Layer 1: Exact / substring ---
            if skill_lower in resume_lower:
                matched.append((skill, 1.0))
                continue

            # --- Layer 2: Fuzzy ---
            fuzzy_score = fuzz.partial_ratio(skill_lower, resume_lower)
            if fuzzy_score >= self.FUZZY_THRESHOLD:
                matched.append((skill, round(fuzzy_score / 100, 2)))
                continue

            # --- Layer 3: Semantic ---
            if sent_embeddings is not None:
                try:
                    skill_emb = self.ml.encode([skill])
                    sims = self.ml.batch_similarity(skill_emb, sent_embeddings)
                    max_sim = float(sims.max())
                    if max_sim >= self.SEMANTIC_THRESHOLD:
                        matched.append((skill, round(max_sim, 2)))
                        continue
                except Exception:
                    pass

            unmatched.append(skill)

        return matched, unmatched

    def _extract_sentences(self, text: str, max_lines: int = 100) -> List[str]:
        """Return clean non-empty lines capped for performance."""
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 5:
                lines.append(line)
            if len(lines) >= max_lines:
                break
        return lines
