import numpy as np
from typing import List
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from services.documents import DocumentObject

# --- AYARLAR ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 

# Model cevabı bulamadığında döneceği standart mesaj
NO_ANSWER_MSG = "I am sorry, I could not find the answer to this question in the provided documents."

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype=torch.float32,
    trust_remote_code=True
)
model.eval()

# --- YARDIMCI FONKSİYONLAR ---

def calculate_hybrid_match(question: str, text: str) -> float:
    """
    Keyword Skorlayıcı: Stop words temizlenir, özel isimler (Proper Nouns) korunur.
    Özel isimler metinde yoksa skor düşürülür (ceza).
    """
    stops = {
        "what", "how", "why", "when", "does", "do", "did", "can", "could", 
        "use", "using", "used", "code", "file", "make", "create",
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
        "with", "by", "of", "is", "are", "was", "were", "be", "been", "should"
    }
    
    words = [w for w in re.findall(r"\b\w{3,}\b", question.lower()) if w not in stops]
    
    if not words: return 0.5 
    
    # Kelime Ağırlıkları
    weights = {}
    question_words_original = re.findall(r"\b\w{3,}\b", question)
    word_case_map = {w.lower(): w for w in question_words_original}

    for w_lower in words:
        original = word_case_map.get(w_lower, w_lower)
        # Baş harfi büyükse (Özel İsim) -> Yüksek Puan
        if original[0].isupper():
            weights[w_lower] = 3.0
        elif len(w_lower) > 6:
            weights[w_lower] = 1.5
        else:
            weights[w_lower] = 1.0
    
    total_weight = sum(weights.values())
    text_lower = text.lower()
    score, penalty = 0.0, 0.0

    for word, weight in weights.items():
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            score += weight
        elif weight >= 3.0: 
            # Özel isim yoksa ceza kes
            penalty += 1.0

    return max(0.0, (score - penalty) / total_weight)

# --- RETRIEVAL (ARAMA) ---

def retrieve_globally_relevant_chunks(
    question: str,
    documents: List[DocumentObject],
    k_per_doc: int = 5,
    max_chunks: int = 5,
    threshold: float = 0.35,
    vec_weight: float = 0.65
) -> List[str]:
    """
    Tüm dokümanlar arasında hem vektör benzerliği hem de anahtar kelime eşleşmesi
    kullanarak en alakalı parçaları bulur (Hybrid Search).
    """
    candidates = list({r["text"] for doc in documents for r in doc.embedding_store.search(question, k=k_per_doc)})
    if not candidates: return []

    emb_model = documents[0].embedding_store.model
    q_vec = emb_model.encode([question], convert_to_numpy=True)
    c_vecs = emb_model.encode(candidates, convert_to_numpy=True)
    
    v_scores = (c_vecs @ q_vec.T).squeeze() / (np.linalg.norm(c_vecs, axis=1) * np.linalg.norm(q_vec))
    if v_scores.ndim == 0: v_scores = [v_scores]

    final_results = []
    
    for text, v_score in zip(candidates, v_scores):
        k_score = calculate_hybrid_match(question, text)
        # Vektör ve Keyword skorlarını ağırlıklandırarak birleştir
        h_score = (v_score * vec_weight) + (k_score * (1 - vec_weight))
        
        if h_score >= threshold:
            final_results.append((h_score, text))

    # En yüksek skorlu parçaları döndür
    return [res[1] for res in sorted(final_results, key=lambda x: x[0], reverse=True)[:max_chunks]]

# --- GENERATION (CEVAP ÜRETME) ---

def generate_answer_from_contexts(
    question: str,
    contexts: List[str]
) -> str:
    """
    Bulunan bağlamları kullanarak LLM'e (Language Model) cevap ürettirir.
    """
    if not contexts:
        return NO_ANSWER_MSG

    context_text = "\n\n".join(contexts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. Your task is to answer the user's question based strictly on the provided context.\n"
                "Steps to follow:\n"
                "1. Read the context carefully.\n"
                "2. Find the specific sentences that answer the question.\n"
                "3. Synthesize the answer in a clear, readable format.\n"
                f"If the context does not contain the answer, reply exactly with: '{NO_ANSWER_MSG}'"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion:\n{question}"
        }
    ]

    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # --- TEMİZLİK ---
    
    # 1. Modelin kendi kendine konuşmasını temizle
    for cut_phrase in ["Answer:", "Explanation:", "Human:", "User:", "Note:"]:
        if cut_phrase in answer:
            answer = answer.replace(cut_phrase, "").strip()

    # 2. Hatalı placeholder çıktısını düzelt
    if "{NO_ANSWER_MSG}" in answer or "NO_ANSWER_MSG" in answer:
        return NO_ANSWER_MSG

    # 3. Telif hakkı vb. uyarıları varsa cevap yoktur (Hallüsinasyon önlemi)
    for banned in ["Packt", "Publishing", "Copyright", "All rights reserved"]:
        if banned.lower() in answer.lower():
            return NO_ANSWER_MSG
            
    # 4. Cevap çok kısaysa (anlamsızsa)
    if len(answer) < 10:
        return NO_ANSWER_MSG

    return answer