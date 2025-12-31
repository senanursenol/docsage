from typing import List
from services.embedding_service import EmbeddingStore

def split_into_chunks(
    text: str,
    max_chars: int = 500,
    overlap: int = 100
):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())

        start = end - overlap
        if start < 0:
            start = 0

    return chunks

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
