import streamlit as st
import os
from engine import RAGEngine

st.set_page_config(page_title="AI RAG Chatbot", layout="wide")

st.title("🤖 AI RAG Document Assistant")
st.markdown("Upload documents in the sidebar and ask questions from them.")

# ---------------- SESSION STATE ----------------
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()

# ---------------- FILE STORAGE FOLDER ----------------
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- SIDEBAR UPLOAD ----------------
with st.sidebar:
    st.header("📂 Upload Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:

            # prevent reprocessing same file
            if uploaded_file.name in st.session_state.uploaded_files:
                continue

            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Indexing {uploaded_file.name} ..."):
                try:
                    st.session_state.rag_engine.load_documents(save_path)
                    st.session_state.uploaded_files.add(uploaded_file.name)
                    st.success(f"Indexed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed: {uploaded_file.name}")
                    st.error(e)

# ---------------- DISPLAY CHAT HISTORY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ----------------
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking... 🔎"):
                response = st.session_state.rag_engine.query(prompt)

            st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            st.error(f"Error: {e}")

            if "GROQ_API_KEY" in str(e) or "401" in str(e):
                st.info("Add GROQ_API_KEY inside your .env file.")