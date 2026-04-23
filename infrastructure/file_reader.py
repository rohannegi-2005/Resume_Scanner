import os
import pdfplumber
from docx import Document


class FileReader:
    SUPPORTED = {".pdf", ".docx"}

    def read(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._read_pdf(file_path)
        elif ext == ".docx":
            return self._read_docx(file_path)
        raise ValueError(f"Unsupported file type: {ext}")

    def _read_pdf(self, path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            raise RuntimeError(f"PDF read error: {e}")
        return text.strip()

    def _read_docx(self, path: str) -> str:
        try:
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            paragraphs.append(cell.text.strip())
            return "\n".join(paragraphs).strip()
        except Exception as e:
            raise RuntimeError(f"DOCX read error: {e}")
