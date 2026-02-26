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
    uploaded_files = st.file_uploader("Upload PDF or Text files", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for uploaded_file in uploaded_files:
                    tmp_file_path = os.path.join(tmp_dir, uploaded_file.name)
                    with open(tmp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    try:
                        st.session_state.rag_engine.load_documents(tmp_file_path)
                        st.success(f"Loaded: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error loading {uploaded_file.name}: {e}")

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
            # We want to show sources right away but stream the text
            # To do that with RunnableParallel, we need to iterate over the streaming chunks
            answer_placeholder = st.empty()
            full_answer = ""
            sources = []
            
            # Use the streaming generator
            for chunk in st.session_state.rag_engine.query(prompt):
                if "answer" in chunk:
                    full_answer += chunk["answer"]
                    answer_placeholder.markdown(full_answer + "▌")
                if "context" in chunk:
                    sources = chunk["context"]
            
            # Remove the cursor
            answer_placeholder.markdown(full_answer)
            
            # Render sources
            source_names = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in sources]))
            source_text = "\n\n**Sources:** " + ", ".join(source_names) if source_names else ""
            
            full_response = full_answer + source_text
            
            # Re-render with sources
            answer_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error: {e}")
            if "GROQ_API_KEY" in str(e) or "401" in str(e):
                st.info("Please make sure your GROQ_API_KEY is correctly set in the .env file.")
