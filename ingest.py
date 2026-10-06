import os
# کد قبلی:
# from langchain_community.document_loaders import TextLoader, DirectoryLoader
# کد اصلاح شده:
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# 1. پیکربندی اتصال به سرویس‌ها (طبق docker-compose)
QDRANT_URL = "http://localhost:6333"
#QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # ← این خط اضافه شود
QDRANT_API_KEY ="992b61d895665cd63484456f34950944dd181ea656ed5e2bf2e4268dae4e96d8"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "assistant_knowledge1"  # نام کالکشن در Qdrant

# 2. بارگذاری مستندات (مثال: همه فایل‌های docx در پوشه docs)
# برای PDF/DOCX باید از Loaderهای مربوطه مثل PyPDFLoader استفاده کنید

# تغییر در نحوه بارگذاری:
#loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
loader = DirectoryLoader("./docs", glob="**/*.docx",  # فقط فایل‌های Word
    loader_cls=UnstructuredWordDocumentLoader)
documents = loader.load()

if not documents:
    print("هیچ سندی برای ایندکس پیدا نشد!")
    exit()

# 3. تکه‌تکه کردن متون (Chunking)
# طبق منابع، اندازه 1000 کاراکتر با 200 کاراکتر همپوشانی نقطه شروع خوبی است
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
chunks = text_splitter.split_documents(documents)
print(f"تعداد {len(chunks)} تکه متن برای ایندکس آماده شد.")

# 4. تعریف مدل Embedding محلی
# nomic-embed-text یک مدل سبک و بسیار مناسب برای RAG است
embeddings = OllamaEmbeddings(
    #model="nomic-embed-text",
    model="qllama/bge-m3:q4_k_m",
    #model="aligh4699/heydariAI-persian-embeddings",
    #model="partai/dorna-llama3:8b-instruct-q4_0",
    base_url=OLLAMA_URL
)

# 5. تشخیص خودکار ابعاد از روی مدل (بهترین روش، دیگر دستی نمی‌نویسید)
sample_vector = embeddings.embed_query("test")
VECTOR_SIZE = len(sample_vector)
print(f"ابعاد بردار مدل Embedding: {VECTOR_SIZE}")


# 6. ایجاد/اتصال به کالکشن Qdrant و ذخیره بردارها
#client = QdrantClient(url=QDRANT_URL)
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,  # ← این پارامتر اضافه شود
)

# 7. حذف کالکشن قبلی (اگر وجود دارد)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print("کالکشن قبلی حذف شد.")

# 8 بررسی اینکه کالکشن وجود دارد یا خیر و ایجاد آن
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
print(f"کالکشن جدید با ابعاد {VECTOR_SIZE} ساخته شد.")

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# 9 افزودن تکه‌ها به دیتابیس برداری
vector_store.add_documents(chunks)
print("✅ ایندکس‌گذاری با موفقیت انجام شد.")
