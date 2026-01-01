from langchain_huggingface import HuggingFaceEmbeddings
import math

class TextSimilarityEngine:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Load model ONLY ONCE
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def _embed(self, text: str):
        """
        Convert text to vector
        """
        return self.embeddings.embed_query(text)

    def _cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def similarity(self, text1: str, text2: str):
        """
        High-level API: text → similarity score
        """
        vec1 = self._embed(text1)
        vec2 = self._embed(text2)
        return self._cosine_similarity(vec1, vec2)