from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from routers.documents import DOCUMENT_STORE
# retrieve fonksiyonunun adını ve dönüş tipini serviste güncellediğimizi varsayarak (aşağıda anlatacağım):
from services.qa_service import generate_answer_from_contexts, retrieve_globally_relevant_chunks

router = APIRouter(prefix="/qa", tags=["qa"])

class QuestionRequest(BaseModel):
    doc_ids: List[str]
    question: str

class QAResponse(BaseModel):
    answer: str
    context_chunks: List[str]

@router.post("/", response_model=QAResponse)
def qa_endpoint(request: QuestionRequest):
    documents = []
    for doc_id in request.doc_ids:
        if doc_id not in DOCUMENT_STORE:
            raise HTTPException(status_code=404, detail=f"Doküman bulunamadı: {doc_id}")
        documents.append(DOCUMENT_STORE[doc_id])

    # 1. Semantik Arama (Retrieval)
    # Burada servisten sadece metinleri değil, skorları da dikkate alacak bir yapı kurabiliriz.
    # Şimdilik standart retrieval yapalım.
    contexts = retrieve_globally_relevant_chunks(
        question=request.question,
        documents=documents,
        k_per_doc=5,
        max_chunks=5
    )

    # Eğer hiç bağlam bulunamadıysa veya bulunan bağlamlar çok kısaysa direkt red.
    if not contexts or len(" ".join(contexts).strip()) < 50:
        return QAResponse(
            answer="Bu dokümanlarda bu bilgi yer almıyor (İlgili içerik bulunamadı).",
            context_chunks=[]
        )

    # 2. LLM ile Cevap Üretme
    # Gereksiz "focus_tokens" kontrolünü sildik. Artık semantik bağlam çalışacak.
    # LLM, prompt içindeki "Bilgi yoksa söyle" talimatına uyacaktır.
    answer = generate_answer_from_contexts(
        question=request.question,
        contexts=contexts
    )

    return QAResponse(
        answer=answer,
        context_chunks=contexts
    )