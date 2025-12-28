from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
import uuid

from services.document_service import extract_text_from_file
from services.documents import DocumentObject
from services.qa_service import split_into_chunks

router = APIRouter(prefix="/documents", tags=["documents"])

DOCUMENT_STORE: Dict[str, DocumentObject] = {}

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "document"
    ext = filename.split(".")[-1].lower()

    if ext not in ["pdf", "docx"]:
        raise HTTPException(status_code=400, detail="Sadece PDF ve DOCX destekleniyor.")

    content = await file.read()
    text = extract_text_from_file(content, ext)

    chunks = split_into_chunks(text)
    doc_obj = DocumentObject(chunks=chunks)

    doc_id = str(uuid.uuid4())
    DOCUMENT_STORE[doc_id] = doc_obj

    return {
        "doc_id": doc_id,
        "filename": filename,
        "num_chars": len(chunks)
    }
