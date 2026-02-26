from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine import RAGEngine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Chatbot API is running"}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        if rag_engine.vectorstore is None:
            # Check if there's a persisted DB
            import os
            if os.path.exists("./chroma_db"):
                from langchain_community.vectorstores import Chroma
                rag_engine.vectorstore = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=rag_engine.embeddings
                )
            else:
                return {"answer": "I don't have any data yet. Please upload documents through the management UI."}
        
        answer = rag_engine.query(request.prompt)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
