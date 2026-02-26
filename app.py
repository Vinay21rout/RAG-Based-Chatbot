import streamlit as st
import os
from engine import RAGEngine
import tempfile

st.set_page_config(page_title="AI RAG Chatbot", layout="wide")

st.title("🤖 Minimal AI RAG Chatbot")
st.markdown("---")

if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Knowledge")
    uploaded_file = st.file_uploader("Upload PDF or Text file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                st.session_state.rag_engine.load_documents(tmp_file_path)
                st.success(f"Loaded: {uploaded_file.name}")
                os.remove(tmp_file_path)
            except Exception as e:
                st.error(f"Error loading file: {e}")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.rag_engine.query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")
            if "GROQ_API_KEY" in str(e) or "401" in str(e):
                st.info("Please make sure your GROQ_API_KEY is correctly set in the .env file.")
