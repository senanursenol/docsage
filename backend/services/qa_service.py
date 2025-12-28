import numpy as np

from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

from services.embedding_service import EmbeddingStore
from services.documents import DocumentObject

embedding_store = EmbeddingStore()

MODEL_NAME = "google/flan-t5-base" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

embedding_store = EmbeddingStore()

from typing import List, Tuple

def clean_answer(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text

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

def generate_answer_from_contexts(
    question: str,
    contexts: List[str]
) -> str:
    context_text = "\n\n".join(contexts)

    prompt_2 = f"""
    Role:
    You are a document-grounded question answering expert. Your task is to answer questions strictly and exclusively based on the provided document text.

    Core Principle:
    - The document text is the ONLY source of truth.
    - Every part of your answer must be directly supported by the document content.

    Rules:
    1. Use only the information explicitly stated or clearly implied within the document.
    2. You may reason semantically within the document (understand meaning, paraphrase, and connect related statements), but you must NOT introduce new facts.
    3. Do NOT use any external knowledge, prior training data, or assumptions.
    4. The answer must be a single, short explanatory sentence.
    5. Do NOT include book titles, author names, copyright notices, or publication details.
    6. Do NOT add explanations, justifications, examples, or restate the question.
    7. Do NOT mention the document, source, or analysis process.

    Fallback Rule:
    - If the document does NOT contain enough information to answer the question, output EXACTLY the following sentence and nothing else:
    "Bu dokümanlarda bu bilgi yer almıyor."

    Document Text:
    {context_text}

    Question:
    {question}

    Output Format:
    Answer:
    """

    inputs = tokenizer(
        prompt_2,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_beams=4,
        early_stopping=True,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    for banned in ["Packt", "Publishing", "Edition", "Copyright"]:
        if banned.lower() in answer.lower():
            return "Bu dokümanlarda bu bilgi yer almıyor."
        
    return answer

def retrieve_from_multiple_documents(
    question: str,
    documents: List[DocumentObject],
    k_per_doc: int = 3,
    max_chunks: int = 5
) -> List[str]:
    all_chunks = []

    for document in documents:
        results = document.embedding_store.search(
            question,
            k=k_per_doc
        )
        for r in results:
            all_chunks.append(r["text"])

    # şimdilik: ilk max_chunks tanesini al
    return all_chunks[:max_chunks]

def retrieve_globally_relevant_chunks(
    question: str,
    documents: List[DocumentObject],
    k_per_doc: int = 5,
    max_chunks: int = 5
) -> List[str]:
    """
    Her dokümandan aday chunk'lar alır, sonra soru ile
    GLOBAL similarity'ye göre sıralayıp en iyileri döndürür.
    """
    candidates = []

    for document in documents:
        # Doküman bazlı adayları al
        results = document.embedding_store.search(question, k=k_per_doc)
        for r in results:
            candidates.append(r["text"])

    if not candidates:
        return []

    # GLOBAL relevance: soru embedding'i ile tüm adayları karşılaştır
    # (aynı embedding modelini kullanıyoruz)
    emb_model = documents[0].embedding_store.model
    q_emb = emb_model.encode([question], convert_to_numpy=True)
    c_embs = emb_model.encode(candidates, convert_to_numpy=True)

    # cosine similarity
    q_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    c_norm = c_embs / np.linalg.norm(c_embs, axis=1, keepdims=True)
    sims = np.dot(c_norm, q_norm.T).squeeze()

    # en alakalıları seç
    ranked_idx = np.argsort(-sims)
    top_texts = [candidates[i] for i in ranked_idx[:max_chunks]]

    return top_texts