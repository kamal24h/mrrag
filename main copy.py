# main.py
import os
import uuid
import shutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

# ماژول‌های دیتابیس و ریدیس
from database import Base, engine, get_db, Conversation, Message
from redis_client import get_redis_client, ConversationCache
# ایمپورت تابع Ingest
from ingest_service import run_ingest

# ساخت جداول دیتابیس در صورت عدم وجود
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="Hybrid Assistant RAG API", lifespan=lifespan)

# تنظیمات پایه
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "992b61d895665cd63484456f34950944dd181ea656ed5e2bf2e4268dae4e96d8")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "assistant_knowledge1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qllama/bge-m3:q4_k_m")

# ایجاد ارتباط با Qdrant و LLM
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

def get_rag_chain():
    """زنجیره را داینامیک می‌خواند تا پس از هربار Ingest کالکشن جدید بارگذاری شود"""
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.2)

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

    return (
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
        chain = get_rag_chain()
        answer = chain.invoke(request.prompt)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Pydantic Schemas ---
class BeginConversationRequest(BaseModel):
    appId: str
    tenantId: str
    userId: str

class QueryConversationRequest(BaseModel):
    conversationId: str
    prompt: str

class EndConversationRequest(BaseModel):
    conversationId: str

# ----------------- اندپوینت‌های مکالمه -----------------

@app.post("/beginConversation")
async def begin_conversation(req: BeginConversationRequest, db: Session = Depends(get_db)):
    try:
        conv_id = uuid.uuid4()

        # ۱. ذخیره در PostgreSQL
        db_conv = Conversation(
            id=conv_id,
            app_id=req.appId,
            tenant_id=req.tenantId,
            user_id=req.userId,
            status="active"
        )
        db.add(db_conv)
        db.commit()

        # ۲. ذخیره متادیتا در Redis
        r = get_redis_client()
        ConversationCache.create_conversation(
            redis_client=r,
            conv_id=str(conv_id),
            app_id=req.appId,
            tenant_id=req.tenantId,
            user_id=req.userId
        )

        return {
            "conversationId": str(conv_id),
            "status": "created",
            "appId": req.appId,
            "tenantId": req.tenantId,
            "userId": req.userId
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to begin conversation: {str(e)}")

@app.post("/query")
async def query_assistant(request: QueryConversationRequest):
    """انجام پرسش با بررسی وضعیت در Redis و ثبت پیام در حافظه موقت"""
    r = get_redis_client()
    meta = ConversationCache.get_conversation_meta(r, request.conversationId)
    if not meta or meta.get("status") != "active":
        raise HTTPException(status_code=404, detail="مکالمه یافت نشد یا پایان یافته است.")

    try:
        # ۱. ثبت پیام کاربر در Redis
        ConversationCache.append_message(r, request.conversationId, role="user", content=request.prompt)

        # ۲. تولید پاسخ از RAG
        chain = get_rag_chain()
        answer = chain.invoke(request.prompt)

        # ۳. ثبت پاسخ دستیار در Redis
        ConversationCache.append_message(r, request.conversationId, role="assistant", content=answer)

        return {
            "conversationId": request.conversationId,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/endConversation")
async def end_conversation(req: EndConversationRequest, db: Session = Depends(get_db)):
    """انتقال کل پیام‌ها از ریدیس به Postgres و بستن وضعیت مکالمه"""
    r = get_redis_client()
    meta = ConversationCache.get_conversation_meta(r, req.conversationId)

    db_conv = db.query(Conversation).filter(Conversation.id == req.conversationId).first()
    if not db_conv:
        raise HTTPException(status_code=404, detail="مکالمه در پایگاه داده یافت نشد.")

    # دریافت پیام‌های نگه‌داری شده در Redis
    cached_messages = ConversationCache.get_messages(r, req.conversationId)

    try:
        # ذخیره کل تاریخچه پیام‌ها در جدول Messages
        for msg in cached_messages:
            db_msg = Message(
                conversation_id=db_conv.id,
                role=msg["role"],
                content=msg["content"]
            )
            db.add(db_msg)

        db_conv.status = "closed"
        db_conv.ended_at = datetime.utcnow()
        db.commit()

        # پاک کردن کش مکالمه از ریدیس
        ConversationCache.clear_conversation(r, req.conversationId)

        return {
            "status": "success",
            "message": "مکالمه پایان یافت و داده‌ها در پایگاه داده اصلی آرشیو شدند.",
            "total_messages_saved": len(cached_messages)
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")

# ----------------- اندپوینت‌های سیستم و Ingest -----------------

@app.get("/health")
async def health():
    return {"status": "healthy", "llm_model": LLM_MODEL}

@app.post("/ingest")
async def trigger_ingest():
    """ایندکس کردن کلیه فایل‌های موجود در پوشه docs"""
    try:
        return run_ingest(docs_dir="./docs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/upload-and-ingest")
async def upload_and_ingest(file: UploadFile = File(...)):
    """آپلود فایل جدید docx یا pdf و اجرای فرآیند ایندکس"""
    allowed_extensions = (".docx", ".pdf")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"فرمت فایل نامعتبر است. تنها فرمت‌های مجاز: {allowed_extensions}"
        )

    upload_dir = "./docs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = run_ingest(docs_dir=upload_dir)
        return {
            "uploaded_file": file.filename,
            "ingest_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")