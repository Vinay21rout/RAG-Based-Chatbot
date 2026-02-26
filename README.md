# RAG-Based AI Chatbot 🤖

This is the backend and AI engine for a Retrieval-Augmented Generation (RAG) chatbot. It uses LangChain Expression Language (LCEL), Groq (LLaMA-3), and ChromaDB to read your PDFs and Text files and answer questions based on their content.

## Features
- **Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally for absolute privacy during the embedding phase.
- **Lightning Fast AI**: Uses Groq's high-speed API to run `llama-3.3-70b-versatile`.
- **Admin UI**: Includes a Streamlit user interface (`app.py`) for uploading documents and testing the bot's responses.
- **REST API**: Includes a FastAPI bridge (`api.py`) so external websites (like an e-commerce platform) can talk to the AI.
- **LCEL Architecture**: Built using modern LangChain 0.3+ Expression Language for maximum stability and speed.

## Prerequisites
- **Python 3.11**: Required for stable Pydantic and ChromaDB compatibility.
- **uv**: We recommend using `uv` for lightning-fast environment setup. (`pip install uv`)
- **Groq API Key**: You need a free API key from [Groq](https://console.groq.com/).

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vinay21rout/RAG-Based-Chatbot.git
   cd RAG-Based-Chatbot
   ```

2. **Set up the virtual environment:**
   We strictly recommend Python 3.11 to avoid ChromaDB `nofile` configuration errors.
   ```bash
   uv venv --python=3.11
   uv pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root folder and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### 1. Document Management UI (Streamlit)
To upload PDFs/TXT files and chat with the bot locally to test the RAG performance:
```bash
uv run streamlit run app.py
```

### 2. External API Service (FastAPI)
To expose the RAG engine so other websites can query it:
```bash
uv run uvicorn api:app --port 8000
```
*You can then send POST requests to `http://localhost:8000/chat` with `{"prompt": "your question"}`.*
