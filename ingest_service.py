
import os
import uuid
from typing import List, Dict, Any
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "992b61d895665cd63484456f34950944dd181ea656ed5e2bf2e4268dae4e96d8")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "assistant_knowledge1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qllama/bge-m3:q4_k_m")

def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_collection_exists(client: QdrantClient, embeddings: OllamaEmbeddings):
    if not client.collection_exists(COLLECTION_NAME):
        sample_vector = embeddings.embed_query("init")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=len(sample_vector), distance=Distance.COSINE),
        )

def process_and_index_document(
    file_path: str,
    doc_id: str,
    doc_name: str,
    doc_type: str,
    department: str,
    app_id: str,
    metadata: Dict[str, Any]
) -> int:
    """پردازش یک فایل و ذخیره تکه‌های آن همراه با متادیتای اختصاصی سند"""
    
    # 1. خواندن فایل بر اساس فرمت
    if doc_type.lower() == "pdf":
        loader = PyPDFLoader(file_path)
    elif doc_type.lower() in ["docx", "doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"نوع سند '{doc_type}' پشتیبانی نمی‌شود.")

    raw_docs = loader.load()
    if not raw_docs:
        return 0

    # 2. خرد کردن متن
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    chunks = text_splitter.split_documents(raw_docs)

    # 3. افزودن متادیتا به تمام Chunkها جهت کوئری و حذف سریع در آینده
    for chunk in chunks:
        chunk.metadata.update({
            "doc_id": doc_id,
            "doc_name": doc_name,
            "department": department,
            "app_id": app_id,
            **(metadata or {})
        })

    # 4. ایندکس در Qdrant بدون حذف کل کالکشن
    client = get_qdrant_client()
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    ensure_collection_exists(client, embeddings)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)

    return len(chunks)

def delete_document_vectors(doc_id: str):
    """حذف تمام بردارهای مرتبط با یک سند بر اساس doc_id"""
    client = get_qdrant_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )
    )