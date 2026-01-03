import pdfplumber
import docx

class FileReader:
    @staticmethod
    def extract_text(file_path):
        try:
            if file_path.endswith(".pdf"):
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted: text += extracted + "\n"
                return text
            
            elif file_path.endswith(".docx"):
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            
            return ""
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""