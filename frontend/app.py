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

# --- CSS STYLING (Modern & Clean) ---
st.markdown("""
<style>
    /* GENERAL BACKGROUND */
    .stApp {
        background-color: #f8f9fa; /* Very light grey/white */
    }
    
    /* HEADERS */
    h1, h2, h3 {
        color: #2c3e50; /* Dark Blue-Grey */
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #eef2f5;
        border-right: 1px solid #d1d1e0;
    }

    /* CHAT MESSAGES */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e1e4e8;
    }
    
    /* USER AVATAR */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #2c3e50 !important;
        color: white !important;
    }

    /* ASSISTANT AVATAR */
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #27ae60 !important; /* Emerald Green */
        color: white !important;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #bdc3c7;
    }

    /* BUTTONS */
    .stButton button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #34495e;
        color: white;
        border-color: #34495e;
    }
    
    /* INFO BOX STYLING */
    .stInfo {
        background-color: #e8f6f3;
        border-left-color: #27ae60;
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
    """Resets memory when a new file is uploaded."""
    st.session_state.doc_id = None
    st.session_state.messages = [] 

# --- SIDEBAR (History & Actions) ---
with st.sidebar:
    st.title("🗂️ History")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    
    if len(st.session_state.history) > 0:
        st.markdown("### Recent Questions")
        # Show last 5 questions in reverse order
        for i, item in enumerate(reversed(st.session_state.history[-5:])): 
            st.caption(f"❓ **{item['question']}**")
            st.markdown("---")
    else:
        st.info("No questions asked yet.")

# --- MAIN PAGE ---
st.title("🧠 DocSage Assistant")

# ENGLISH ONLY WARNING
st.info(
    "👋 **Welcome!** This system is optimized for **English content**.\n\n"
    "Please upload **English** documents (PDF/DOCX) and ask your questions in **English** for the best accuracy."
)

# 1. FILE UPLOAD SECTION
with st.container():
    uploaded_file = st.file_uploader(
        "📄 Upload English Document (PDF, DOCX) or Image",
        type=ALLOWED_DOC_TYPES + ALLOWED_IMAGE_TYPES,
        key="file_upload",
        on_change=reset_doc_id
    )

    if uploaded_file and st.session_state.doc_id is None:
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

# Image Preview (Optional)
if uploaded_file and uploaded_file.type.startswith("image"):
    with st.expander("🖼️ View Uploaded Image"):
        st.image(uploaded_file, use_column_width=True)

st.markdown("---")

# 2. CHAT INTERFACE
# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 Reference Sources (Evidence)"):
                for source in message["sources"]:
                    st.caption(f"• {source}")

# 3. USER INPUT
placeholder_text = "Ask your question here (e.g., 'What is the main topic?')..."
if prompt := st.chat_input(placeholder_text):
    
    # Check if document is loaded
    if not st.session_state.doc_id:
        st.warning("⚠️ Please upload a document first to start chatting.")
        st.stop()

    # Check for English characters (Basic check - Optional)
    # This is a soft reminder, not a blocker.
    if any(char in prompt.lower() for char in "ğşüöçı"):
        st.toast("💡 Tip: Using English will provide better results.", icon="🇺🇸")

    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Backend Request
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

        # --- STREAMING EFFECT & ERROR HANDLING ---
        
        # Keywords indicating the AI couldn't find an answer (Matches Backend NO_ANSWER_MSG)
        error_keywords = ["i am sorry", "could not find", "no information found"]
        
        is_negative_answer = any(keyword in answer_text.lower() for keyword in error_keywords)

        if is_negative_answer:
            # Yellow warning box for negative answers
            st.warning(f"⚠️ {answer_text}")
            full_response = answer_text 
        else:
            # Normal streaming for positive answers
            for chunk in answer_text.split(" "): 
                full_response += chunk + " "
                time.sleep(0.05) # Typing speed
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)

            # Show Sources only for positive answers
            if sources:
                with st.expander("🔍 Reference Sources (Evidence)"):
                    for src in sources:
                        clean_src = src.replace("\n", " ").strip()
                        st.info(f"📄 ...{clean_src[:250]}...") 

        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "sources": sources if not is_negative_answer else []
        })
        
        st.session_state.history.append({"question": prompt, "answer": full_response})