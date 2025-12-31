import numpy as np

from typing import List
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from services.documents import DocumentObject

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

model.eval()

def is_negative_question(question: str) -> bool:
    """
    Belgenin bir konuyu içerip içermediğini soran
    evet/hayır tipi soruları yakalar.
    """
    if not isinstance(question, str):
        return False

    q = question.lower().strip()

    patterns = [
        r"^does the document mention\b",
        r"^does it mention\b",
        r"^does it talk about\b",
        r"^is there any mention of\b",
        r"^is .* discussed\b",
        r"^is .* mentioned\b",
    ]

    return any(re.search(p, q) for p in patterns)

def extract_question_focus(question: str) -> str | None:
    """
    Negatif (does/is) sorularda sorulan asıl kavramı çıkarır.

    Örnek:
    "Does the document mention design patterns?"
    -> "design patterns"
    """
    if not isinstance(question, str):
        return None

    q = question.lower().strip()

    patterns = [
        r"does the document mention (.+)",
        r"does it mention (.+)",
        r"does it talk about (.+)",
        r"is there any mention of (.+)",
        r"is (.+) discussed",
        r"is (.+) mentioned",
    ]

    for p in patterns:
        m = re.search(p, q)
        if m:
            return m.group(1).strip(" ?.")

    return None

def generate_answer_from_contexts(
    question: str,
    contexts: List[str]
) -> str:
    context_text = "\n\n".join(contexts)

    prompt_2 = f"""
    You are a document-grounded question answering assistant.

    Your task is to answer the question using ONLY the information provided in the document text below.
    You may paraphrase and combine related statements from the document, but you MUST NOT add any new information or external knowledge.

    Rules:
    - The answer must be a single, clear, explanatory sentence.
    - Do NOT include book titles, author names, copyright notices, or publication details.
    - Do NOT add examples, explanations, or restate the question.
    - Do NOT mention the document or your reasoning process.

    If the document does NOT contain enough information to answer the question, reply EXACTLY with:
    "Bu dokümanlarda bu bilgi yer almıyor."

    Document:
    {context_text}

    Question:
    {question}

    Answer:
    """

    inputs = tokenizer(
        prompt_2,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "Answer:" in decoded:
        answer = decoded.split("Answer:")[-1].strip()
    else:
        answer = decoded.strip()

    # Güvenlik: çok kısa / anlamsız cevap
    if len(answer) < 10:
        return "Bu dokümanlarda bu bilgi yer almıyor."

    # Metadata sızıntı guard
    for banned in ["Packt", "Publishing", "Edition", "Copyright"]:
        if banned.lower() in answer.lower():
            return "Bu dokümanlarda bu bilgi yer almıyor."
        
    return answer

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