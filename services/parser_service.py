import re
from dateutil import parser
from datetime import datetime

class ParserService:
    
    @staticmethod
    def calculate_experience_score(text):
        text = text.lower()
        experience_section = ""
        headings = ["experience", "work experience", "professional experience", "employment history"]
        
        # 1. Extract Section
        lines = text.split("\n")
        capturing = False
        for line in lines:
            clean_line = line.strip().lower()

            if clean_line.rstrip(":") in headings:
                capturing = True
                continue
            elif capturing and (clean_line == "" or clean_line.endswith(":")):
                break
            elif capturing:
                experience_section += line + "\n"

        # If extraction failed, scan whole text (optional fallback)
        target_text = experience_section if experience_section else text

        # 2. Extract Dates
        patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*[-to]+\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|present)',
            r'(\d{1,2}[-/]\d{2,4})\s*[-to]+\s*(\d{1,2}[-/]\d{2,4}|present)',
            r'([a-z]+\s\d{4})\s*[-to]+\s*([a-z]+\s\d{4}|present)'
        ]
        
        total_months = 0
        for pat in patterns:
            matches = re.findall(pat, target_text)
            for start, end in matches:
                try:
                    s_date = parser.parse(start, dayfirst=True)
                    e_date = datetime.now() if "present" in end else parser.parse(end, dayfirst=True)
                    months = (e_date.year - s_date.year) * 12 + (e_date.month - s_date.month)
                    total_months += max(0, months)
                except:
                    continue

        # 3. Score Rules
        if total_months >= 60: return 100
        if total_months >= 36: return 70
        if total_months >= 24: return 50
        if total_months >= 12: return 25
        return 0

    @staticmethod
    def calculate_qualification_score(text, required_degrees):
        # Simple extraction
        text = text.lower()
        # Find matches
        for degree in required_degrees:
            if degree.lower().strip() in text:
                return 100
        return 0