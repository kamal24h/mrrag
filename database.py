
import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

POSTGRES_USER = os.getenv("POSTGRES_USER", "assistant_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "Arian@123")
POSTGRES_DB = os.getenv("POSTGRES_DB", "assistant_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_name = Column(String(255), nullable=False)
    doc_type = Column(String(50), nullable=False)        # pdf, docx, etc.
    department = Column(String(100), nullable=False, index=True)
    app_id = Column(String(100), nullable=False, index=True)
    meta_info = Column(JSONB, nullable=True)             # متادیتاهای دلخواه JSON
    chunks_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    app_id = Column(String(100), nullable=False, index=True)
    tenant_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    status = Column(String(20), default="active")  # active, closed
    created_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()