import streamlit as st
import requests
from PIL import Image

# --- SABİTLER ---
BACKEND_URL = "http://localhost:8000/api/doc-vqa"
ALLOWED_DOC_TYPES = ["pdf", "docx"]
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]

st.set_page_config(
    page_title="DocSage - Akıllı İçerik Sorgulama",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GEÇMİŞ PANELİ ---
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title("📜 Geçmiş Sorular")
    if len(st.session_state.history) > 0:
        for i, item in enumerate(reversed(st.session_state.history[-10:])):
            st.markdown(f"**{i+1}. {item['question']}**")
            st.caption(f"🧠 {item['answer']}")
            st.markdown("---")
    else:
        st.info("Henüz bir geçmiş bulunmuyor.")
    if st.button("🗑️ Geçmişi Temizle"):
        st.session_state.history = []
        st.rerun()

# --- ARAYÜZ BAŞLIK ---
st.title("🧠 DocSage: Akıllı İçerik Sorgulama Sistemi")
st.markdown("PDF, Word veya görsel yükleyerek doğal dilde anlık yanıt alın.")
st.markdown("---")

# --- DOSYA YÜKLEME ALANI ---
st.subheader("📎 Belge veya Görsel Ekle")

uploaded_file = st.file_uploader(
    "Dosya veya Görsel Yükle (PDF, DOCX, JPG, PNG):",
    type=ALLOWED_DOC_TYPES + ALLOWED_IMAGE_TYPES,
    key="file_upload"
)

# Önizleme
uploaded_image = None
if uploaded_file:
    if uploaded_file.type.startswith("image"):
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Yüklenen Görsel", use_column_width=True)
    else:
        st.success(f"📎 Yüklendi: {uploaded_file.name}")

# --- SORU ALANI (METİN KUTUSU ENTEGRE DOSYA) ---
st.markdown("---")
st.subheader("💬 Sorunuzu Yazın")

st.markdown(
    """
    <style>
    /* GENEL ARKA PLAN */
    .stApp {
        background-color: #d6f7d6;; /* Soft Açık Yeşil */
    }

    section.main {
        padding-top: 2rem; /* Üst boşluğu artır */
    }

    /* Kenar Çubuğu ve Başlık Fontu */
    [data-testid="stSidebar"] {
        background-color: #e6e6fa; /* Açık Lavanta */
        color: #191970; /* Koyu Mavi */
    }
    h1, h2, h3 {
        color: #191970; /* Koyu Mavi Başlıklar */
    }

    /* YÜKLEME KUTULARI VE METİN ALANLARI */
    [data-testid="stFileUploader"], [data-testid="stTextArea"], [data-testid="stButton"] button {
        border-radius: 8px;
        background-color: #d4f1d4; /* Soft Yeşil Buton/Alan Dolgu */
        color: #191970; /* Koyu Mavi Yazı */
    }

    /* BİLGİ/UYARI MESAJLARI */
    .stAlert {
        border-left: 6px solid #6a5acd !important; /* Mor Çizgi */
        border-radius: 4px;
        background-color: #f3f3ff; /* Çok Açık Mor Dolgu */
    }

    /* GÖNDER BUTONU ÖZELLEŞTİRMESİ */
    [data-testid="stButton"] > button {
        font-weight: bold;
        transition: all 0.2s;
        background-color: #191970 !important; /* Koyu Mavi */
        color: white !important;
    }
    [data-testid="stButton"] > button:hover {
        background-color: #000080 !important; /* Hover Koyu Mavi */
    }

    /* YÜKLENEN GÖRSEL BAŞLIĞI */
    .caption {
        font-style: italic;
        color: #6a5acd;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ... Kodun geri kalanı aynı ...

col_a, col_b = st.columns([6, 1])
with col_a:
    question = st.text_area(
        "Sorunuzu yazın:",
        placeholder="Dokümanınızla ilgili soruyu buraya yazın...",
        height=120,
        key="question_input"
    )
with col_b:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("📎", unsafe_allow_html=True)

# --- API İŞLEMİ ---
if st.button("🚀 Yanıt Al", use_container_width=True, type="primary"):

    if not question.strip() and not uploaded_file:
        st.error("Lütfen bir dosya veya metin girin.")
        st.stop()

    data = {'question': question}
    files = None
    input_type = None

    if uploaded_file:
        input_type = 'file'
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    elif question.strip():
        input_type = 'text'
        data['text'] = question

    data['input_type'] = input_type

    with st.spinner("🔎 Analiz yapılıyor, lütfen bekleyin..."):
        try:
            response = requests.post(BACKEND_URL, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "Cevap alınamadı.")
                st.markdown("---")
                st.subheader("✅ Yanıt (DocSage)")
                st.success(answer)
                st.session_state.history.append({"question": question, "answer": answer})
            else:
                st.error(f"Backend hatası: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Backend çalışmıyor veya ulaşılamıyor.")
        except Exception as e:
            st.error(f"Beklenmedik hata: {e}")
