from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY
import pgvector.sqlalchemy
import os

from app.db.database import Base

# Determine if we're running in a test environment
IS_TESTING = "pytest" in os.environ.get("PYTHONPATH", "")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), index=True, default="slack")  # slack, discord, etc.
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    content = Column(Text)
    timestamp = Column(String(50))  # Original timestamp from the source
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # For embeddings (dimension is 768 for nomic-embed-text)
    # Store as JSON array in SQLite for testing
    if IS_TESTING:
        embedding = Column(Text)  # For SQLite in tests
    else:
        embedding = Column(pgvector.sqlalchemy.Vector(768))
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User", back_populates="messages") 