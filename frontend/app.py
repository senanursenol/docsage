import streamlit as st
import requests
import time
from PIL import Image

# --- CONFIGURATION & CONSTANTS ---
BASE_URL = "http://localhost:8000"
ALLOWED_DOC_TYPES = ["pdf", "docx"]
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]

st.set_page_config(
    page_title="DocSage - Smart Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (Fixed Sidebar Button Color & Position) ---
st.markdown("""
<style>
    /* GENERAL BACKGROUND */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* HEADERS */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* SIDEBAR - KOYU LACİVERT */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
        border-right: 1px solid #1a252f;
    }
    
    /* SIDEBAR METİNLERİ - GENEL BEYAZ */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .caption {
        color: #ffffff !important;
    }

    /* SIDEBAR 'CLEAR' BUTONU - ÖZEL AYARLAR */
    /* 1. Buton Kutusu (Beyaz Arka Plan) */
    [data-testid="stSidebar"] .stButton button {
        background-color: #ffffff !important;
        border: 1px solid #ffffff;
        width: 100%;
    }
    
    /* 2. Buton İÇİNDEKİ Yazı Rengi (KESİN LACİVERT) */
    /* p etiketi ve butonun kendisi için renk zorlaması */
    [data-testid="stSidebar"] .stButton button, 
    [data-testid="stSidebar"] .stButton button p {
        color: #2c3e50 !important; 
        font-weight: 800 !important; /* Daha kalın yazı */
    }
    
    /* Hover Efekti */
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #ecf0f1 !important;
        border-color: #bdc3c7;
        transform: translateY(-2px);
    }

    /* HISTORY BAŞLIĞI */
    .history-header {
        font-size: 1.8rem !important;
        font-weight: bold !important;
        color: #ffffff !important;
        margin-bottom: 20px !important;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        padding-bottom: 10px;
    }

    /* MAIN PAGE BUTTONS (Process File) */
    .main .stButton button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
        height: auto;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .main .stButton button:hover {
        background-color: #34495e;
        color: white;
    }
    
    /* CHAT STYLES */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e1e4e8;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        border: 2px dashed #bdc3c7;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE MANAGEMENT ---
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "history" not in st.session_state:
    st.session_state.history = []

def reset_doc_id():
    st.session_state.doc_id = None
    st.session_state.messages = [] 

# --- SIDEBAR (History & Actions) ---
with st.sidebar:
    # 1. Başlık
    st.markdown('<div class="history-header">🗂️ History</div>', unsafe_allow_html=True)
    
    # 2. Geçmiş Sorular
    if len(st.session_state.history) > 0:
        st.markdown("<h3 style='color: #bdc3c7; font-size: 1rem; margin-top: 10px;'>Recent Questions</h3>", unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.history[-5:])): 
            st.caption(f"❓ **{item['question']}**")
            st.markdown("<hr style='margin: 5px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    else:
        st.info("No questions asked yet.")

    # 3. BUTONU EN ALTA İTMEK İÇİN BÜYÜK BOŞLUK
    # Ekran boyutuna göre otomatik boşluk oluşturur
    st.markdown("""
        <style>
            div[data-testid="stSidebar"] > div:first-child {
                height: 100vh;
                display: flex;
                flex-direction: column;
            }
            div[data-testid="stSidebar"] > div:first-child > div:nth-child(2) {
                flex-grow: 1; /* Bu kısım aradaki boşluğu doldurur */
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Bu boş div yukarıdaki flex-grow sayesinde tüm boşluğu kaplayacak
    st.markdown("<div></div>", unsafe_allow_html=True)
    
    # 4. Clear Butonu (En Alta Sabitlenmiş Olacak)
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# --- MAIN PAGE ---

# ORTALANMIŞ BAŞLIK
st.markdown("<h1 style='text-align: center;'>🧠 DocSage Assistant</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# WARNING
st.info(
    "👋 **Welcome!** This system is optimized for **English content**.\n\n"
    "Please upload **English** documents (PDF/DOCX) and ask your questions in **English** for the best accuracy."
)

# FILE UPLOAD SECTION
with st.container():
    uploaded_file = st.file_uploader(
        "📄 Upload English Document (PDF, DOCX) or Image",
        type=ALLOWED_DOC_TYPES + ALLOWED_IMAGE_TYPES,
        key="file_upload",
        on_change=reset_doc_id
    )

    # Process Button
    if st.button("🚀 Process File", use_container_width=True):
        if uploaded_file:
            with st.spinner("⚙️ Analyzing document... Please wait."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BASE_URL}/documents/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.doc_id = data["doc_id"]
                        st.success(f"✅ Document Ready! ID: {data['doc_id']}")
                        time.sleep(1) 
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Connection error: {e}")
        else:
            st.warning("⚠️ Please select a file first.")

# Image Preview
if uploaded_file and uploaded_file.type.startswith("image"):
    with st.expander("🖼️ View Uploaded Image"):
        st.image(uploaded_file, use_column_width=True)

st.markdown("---")

# CHAT INTERFACE
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Reference Sources (Evidence)"):
                for source in message["sources"]:
                    st.caption(f"• {source}")

# USER INPUT
placeholder_text = "Ask your question here (e.g., 'What is the main topic?')..."
if prompt := st.chat_input(placeholder_text):
    
    if not st.session_state.doc_id:
        st.warning("⚠️ Please upload a document first to start chatting.")
        st.stop()

    if any(char in prompt.lower() for char in "ğşüöçı"):
        st.toast("💡 Tip: Using English will provide better results.", icon="🇺🇸")

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []
        
        with st.spinner("Searching knowledge base..."):
            try:
                payload = {"doc_ids": [st.session_state.doc_id], "question": prompt}
                response = requests.post(f"{BASE_URL}/qa/", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    answer_text = result.get("answer", "No answer generated.")
                    sources = result.get("context_chunks", [])
                else:
                    answer_text = f"Error: {response.text}"
            except Exception as e:
                answer_text = f"Connection error: {e}"

        error_keywords = ["i am sorry", "could not find", "no information found"]
        is_negative_answer = any(keyword in answer_text.lower() for keyword in error_keywords)

        if is_negative_answer:
            st.warning(f"⚠️ {answer_text}")
            full_response = answer_text 
        else:
            for chunk in answer_text.split(" "): 
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)

            if sources:
                with st.expander("🔍 Reference Sources (Evidence)"):
                    for src in sources:
                        clean_src = src.replace("\n", " ").strip()
                        st.info(f"📄 ...{clean_src[:250]}...") 

        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "sources": sources if not is_negative_answer else []
        })
        
        st.session_state.history.append({"question": prompt, "answer": full_response})