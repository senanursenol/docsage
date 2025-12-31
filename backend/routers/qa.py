import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from routers.documents import DOCUMENT_STORE
from services.qa_service import (
    generate_answer_from_contexts,
    retrieve_globally_relevant_chunks,
    is_negative_question,
    extract_question_focus
)

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

    # 1️⃣ Retrieval
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

    # 2️⃣ Negatif soru kontrolü 
    question = request.question

    if is_negative_question(question):
        combined_context = " ".join(contexts).lower()

        focus_tokens = [
        w for w in re.findall(r"\b\w+\b", request.question.lower())
        if len(w) > 3 and w not in {"does", "document", "mention", "there", "any"}
    ]

    if not any(tok in combined_context for tok in focus_tokens):
        return QAResponse(
            answer="Bu dokümanlarda bu bilgi yer almıyor.",
            context_chunks=contexts
        )

    # 3️⃣ LLM generation
    answer = generate_answer_from_contexts(
        question=question,
        contexts=contexts
    )

    return QAResponse(
        answer=answer,
        context_chunks=contexts
    )