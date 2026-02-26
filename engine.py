import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Use a global variable to cache the model to avoid reloading in every instance
_embeddings_cache = None

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings_cache

class RAGEngine:
    def __init__(self, persist_directory="vectorstore"):
        self.embeddings = get_embeddings()
        self.persist_directory = persist_directory
        
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
        self.vectorstore = None
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def load_documents(self, file_path):
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        docs = loader.load()
        
        # split text using user's requested settings
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        splits = splitter.split_documents(docs)
        
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(splits)
        
        # Persist the database
        self.vectorstore.persist()
        print("VECTOR DATABASE UPDATED SUCCESSFULLY")

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, user_input):
        if self.vectorstore is None:
            # Try to load existing vectorstore if it exists
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                raise ValueError("Vectorstore not initialized. Load documents first.")
            
        retriever = self.vectorstore.as_retriever()
        
        system_prompt = (
            "You are a helpful AI assistant for an e-commerce website. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
            "Respond in a friendly and professional manner."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # LCEL RAG pipeline
        rag_chain = (
            {"context": retriever | self.format_docs, "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(user_input)
