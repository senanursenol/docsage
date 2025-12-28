from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class EmbeddingStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.ids = []

    def build_index(self, texts: List[str]):
        if not texts:
            raise ValueError("Index oluşturmak için metin yok.")

        embeddings = self.model.encode(texts)

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Geçerli embedding üretilemedi.")

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        # ID yerine index pozisyonunu kullanacağız
        self.texts = texts

    def search(self, query: str, k: int = 5):
        q_emb = self.model.encode([query])
        distances, indices = self.index.search(q_emb, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx]
                })

        return results

      