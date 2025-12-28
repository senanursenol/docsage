from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from routers.documents import DOCUMENT_STORE
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
            raise HTTPException(
                status_code=404,
                detail=f"Doküman bulunamadı: {doc_id}"
            )
        documents.append(DOCUMENT_STORE[doc_id])

    # 1) Multi-document retrieval

    contexts = retrieve_globally_relevant_chunks(
        question=request.question,
        documents=documents,
        k_per_doc=5,
        max_chunks=5
    )

    
    if not contexts or len(" ".join(contexts).strip()) < 50:
        return QAResponse(
            answer="Bu dokümanlarda bu bilgi yer almıyor.",
            context_chunks=[]
        )

    # 2) Generation (tek cevap)
    # answer_question tek DocumentObject alıyordu,
    # burada sadece generation kısmını kullanıyoruz.
    # Bu yüzden contexts'i geçici bir "sahte doküman" gibi veriyoruz.

    # Basit ve güvenli çözüm:
    # contexts'i doğrudan prompt'a verelim
    answer = generate_answer_from_contexts(
    question=request.question,
    contexts=contexts
    )

    # Yukarıdaki çağrıda retrieval kullanılmayacak,
    # prompt'u aşağıda override edeceğiz
    # (küçük ama kontrollü bir hack)

    # Daha temiz çözümü bir sonraki adımda yapacağız.

    return QAResponse(
        answer=answer,
        context_chunks=contexts
    )
