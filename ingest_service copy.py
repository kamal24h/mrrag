# نام فایل: ingest_service.py
import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def run_ingest(docs_dir: str = "./docs"):
    QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "992b61d895665cd63484456f34950944dd181ea656ed5e2bf2e4268dae4e96d8")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "assistant_knowledge1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qllama/bge-m3:q4_k_m")

    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)

    documents = []

    # 1. خواندن فایل‌های Word
    docx_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True
    )
    documents.extend(docx_loader.load())

    # 2. خواندن فایل‌های PDF
    pdf_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents.extend(pdf_loader.load())

    if not documents:
        return {
            "status": "warning",
            "message": "هیچ سند docx یا pdf معتبری در پوشه یافت نشد.",
            "chunks_count": 0
        }

    # loader = DirectoryLoader(docs_dir, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)
    # documents = loader.load()

    # if not documents:
    #     return {"status": "warning", "message": "هیچ سندی با پسوند docx پیدا نشد.", "chunks_count": 0}

    # تکه‌تکه کردن متون
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    
    # اتصال به مدل Embedding
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )

    sample_vector = embeddings.embed_query("test")
    vector_size = len(sample_vector)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # بازسازی کالکشن
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    vector_store.add_documents(chunks)

    return {
        "status": "success",
        "collection_name": COLLECTION_NAME,
        "files_count": len(documents),
        "chunks_count": len(chunks),
        "vector_size": vector_size
    }

if __name__ == "__main__":
    result = run_ingest()
    print(result)