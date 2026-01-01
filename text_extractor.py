import pdfplumber
import docx 
from nltk import ngrams


class TextExtractor:
    def extract_text(self, file_path):
        if file_path.endswith(".pdf"):
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text

        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        return ""
    

    def get_tokens(self, text):
        return text.lower().split()

    def get_phrases(self, tokens, n):
        return [' '.join(gram) for gram in ngrams(tokens, n)]
        