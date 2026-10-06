import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

# --- پیکربندی از طریق Environment Variables با مقادیر پیش‌فرض داکر ---
# در شبکه داکر از نام کانتینر یا نام سرویس استفاده می‌کنیم
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None) # خواندن کلید امنیتی Qdrant
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "assistant_knowledge1")
#LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:8b")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
#EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "aligh4699/heydariAI-persian-embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qllama/bge-m3:q4_k_m")
#EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "partai/dorna-llama3:8b-instruct-q4_0")

app = FastAPI(title="Hybrid Assistant RAG API")

# --- راه‌اندازی سرویس‌ها ---
# ارسال api_key به کلاینت Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.2,
)

# --- پرامپت و زنجیره RAG ---
template = """Answer the question based only on the following context.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class QueryRequest(BaseModel):
    prompt: str

@app.get("/health")
async def health():
    return {"status": "healthy", "llm_model": LLM_MODEL}

@app.post("/query")
async def query_assistant(request: QueryRequest):
    try:
        answer = rag_chain.invoke(request.prompt)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))