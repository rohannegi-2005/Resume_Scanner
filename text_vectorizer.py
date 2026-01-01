from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


    # us class me text ko vector bnana h 
    # or fir ye vector directly semantic matching me use krna h

class TextVectorizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Load model ONLY ONCE
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def embed(self, text: str):
        """
        Convert text to vector
        """
        return self.embeddings.embed_query(text)
    
    def word_to_vec(self, phrases):
        phrase_vectors = []
        for phrase in phrases:
            phrase_vectors.append(self.embed(phrase))
        return phrase_vectors

    