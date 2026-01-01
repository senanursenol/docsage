from typing import List
from services.embedding_service import EmbeddingStore

def split_into_chunks(
    text: str,
    max_chars: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Uzun metni belirtilen karakter sınırına göre parçalara (chunk) böler.
    Bağlam kaybını önlemek için parçalar arasında örtüşme (overlap) bırakır.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())

        # Bir sonraki parçaya geçmek için geri adım at (overlap)
        start = end - overlap
        if start < 0:
            start = 0
            
        # Sonsuz döngü koruması (eğer overlap >= max_chars ise)
        if start >= end:
            start = end

    return chunks

class DocumentObject:
    """
    Tek bir dokümanın işlenmiş halini temsil eder.
    Metin parçalarını (chunks) saklar ve bunlar için vektör veritabanını (EmbeddingStore) yönetir.
    """
    def __init__(self, chunks: List[str]):
        # Çok kısa veya boş parçaları temizle
        cleaned_chunks = [
            c.strip() for c in chunks
            if c and c.strip() and len(c.strip()) > 20
        ]

        if not cleaned_chunks:
            raise ValueError("Dokümandan anlamlı metin çıkarılamadı.")

        self.chunks = cleaned_chunks
        
        # Her dokümanın kendi mini vektör deposu (store) olur
        self.embedding_store = EmbeddingStore()
        self.embedding_store.build_index(cleaned_chunks)