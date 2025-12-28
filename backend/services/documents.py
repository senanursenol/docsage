from typing import List
from services.embedding_service import EmbeddingStore

class DocumentObject:
    def __init__(self, chunks: List[str]):
        cleaned_chunks = [
            c.strip() for c in chunks
            if c and c.strip() and len(c.strip()) > 20
        ]

        if not cleaned_chunks:
            raise ValueError("Dokümandan anlamlı metin çıkarılamadı.")

        self.chunks = cleaned_chunks
        self.embedding_store = EmbeddingStore()
        self.embedding_store.build_index(cleaned_chunks)
