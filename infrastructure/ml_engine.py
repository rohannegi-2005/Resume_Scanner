"""
MLEngine: Replaces the GloVe .kv file approach with sentence-transformers.
- Model: all-MiniLM-L6-v2 (~80 MB, auto-downloads on first run)
- Much better semantic accuracy than GloVe Twitter 200d
- Works on Streamlit Cloud (CPU-only, fits within free tier limits)
- Cached globally so it loads only once per deployment
"""
import numpy as np
from sentence_transformers import SentenceTransformer, util


class MLEngine:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        # Model is loaded via the cached loader in app.py
        # This allows @st.cache_resource to work correctly
        self.model: SentenceTransformer = None

    def set_model(self, model: SentenceTransformer):
        self.model = model

    def encode(self, texts: list):
        """Encode a list of strings into embeddings (tensor)."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    def batch_similarity(self, query_emb, corpus_embs) -> np.ndarray:
        """Return cosine similarity scores between one query and a corpus."""
        sims = util.cos_sim(query_emb, corpus_embs)[0]
        return sims.cpu().numpy()

    def pairwise_similarity(self, emb1, emb2) -> float:
        """Cosine similarity between two single embeddings."""
        return float(util.cos_sim(emb1, emb2).item())
