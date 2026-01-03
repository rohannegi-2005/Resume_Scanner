import os
import numpy as np
from gensim.models import KeyedVectors
from numpy import dot
from numpy.linalg import norm
import streamlit as st  # Used only for caching

class MLEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()

    @st.cache_resource
    def _load_model(_self):
        # The underscore _self tells streamlit not to hash the class instance
        print("⏳ Loading Word2Vec Model... this may take a moment.")
        if not os.path.exists(_self.model_path):
            raise FileNotFoundError(f"Model file not found at {_self.model_path}")
        return KeyedVectors.load(_self.model_path, mmap='r')

    def get_vector(self, text):
        """Returns the average vector for a phrase."""
        words = text.lower().split()
        valid_vectors = [self.model[word] for word in words if word in self.model]
        
        if valid_vectors:
            return np.mean(valid_vectors, axis=0)
        return None

    def cosine_similarity(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))