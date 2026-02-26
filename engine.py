import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------- EMBEDDINGS CACHE ----------------
_embeddings_cache = None

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings_cache


# ---------------- RAG ENGINE ----------------
class RAGEngine:

    def __init__(self, persist_directory="vectorstore"):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()

        # ensure folder exists
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        self.vectorstore = None

        # LLM (Groq)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    # ---------- DOCUMENT INGESTION ----------
    def load_documents(self, file_path):

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        docs = loader.load()

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

        self.vectorstore.persist()
        print("VECTOR DATABASE UPDATED SUCCESSFULLY")

    # ---------- FORMAT DOCUMENTS ----------
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ---------- STEP 1: LLM QUERY PLANNER ----------
    def generate_search_queries(self, question):

        planner_prompt = ChatPromptTemplate.from_template("""
You are a search query planner for a knowledge base.

Your job is to break the user's question into multiple search queries
so that a vector database can retrieve information from different documents.

Rules:
- Create 3 to 5 search queries
- Each query must target a different aspect of the question
- Use short keyword-rich phrases
- Do NOT answer the question
- Do NOT explain anything
- Do NOT number them
- Each query on a new line

User Question:
{question}
""")

        planner_chain = planner_prompt | self.llm | StrOutputParser()

        raw_output = planner_chain.invoke({"question": question})

        queries = [q.strip() for q in raw_output.split("\n") if q.strip()]

        print("\nGenerated Search Queries:")
        for q in queries:
            print(" -", q)

        return queries

    # ---------- STEP 2: MULTI-RETRIEVAL ----------
    def retrieve_context(self, question):

        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k":4})

        queries = self.generate_search_queries(question)

        all_docs = []

        for q in queries:
            docs = base_retriever.invoke(q)
            all_docs.extend(docs)

        # remove duplicate chunks
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

        print("Total retrieved chunks:", len(unique_docs))

        return self.format_docs(unique_docs)

    # ---------- STEP 3: FINAL ANSWERING ----------
    def query(self, user_input):

        # load existing DB if needed
        if self.vectorstore is None:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                raise ValueError("Vectorstore not initialized. Load documents first.")

        # retrieve multi-document context
        context = self.retrieve_context(user_input)

        # final answering prompt
        answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer ONLY using the context provided.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Give a clear and friendly answer.
""")

        answer_chain = answer_prompt | self.llm | StrOutputParser()

        response = answer_chain.invoke({
            "context": context,
            "question": user_input
        })

        return response