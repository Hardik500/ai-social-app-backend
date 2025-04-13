from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import pgvector.sqlalchemy
import os

from app.db.database import Base

# Determine if we're running in a test environment
IS_TESTING = "pytest" in os.environ.get("PYTHONPATH", "")

class PersonalityProfile(Base):
    __tablename__ = "personality_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Personality traits represented as a JSON structure
    traits = Column(JSON)
    
    # Full text description of the personality
    description = Column(Text)
    
    # Vector embedding of the personality description for similarity search
    # Store as Text in SQLite for testing
    if IS_TESTING:
        embedding = Column(Text)  # For SQLite in tests
    else:
        embedding = Column(pgvector.sqlalchemy.Vector(768))
    
    # Whether this is the active profile for the user (users can have multiple profiles)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="personality_profiles")
    
    # All messages used to generate this profile
    message_count = Column(Integer, default=0)
    
    # System prompt to use for personality simulation
    system_prompt = Column(Text)
    
    # Last processed message ID for incremental updates
    last_message_id = Column(Integer, nullable=True)
    
    # Change log for tracking personality changes over time
    change_log = Column(JSON, nullable=True)
    
    # Delta embeddings for incremental changes
    if IS_TESTING:
        delta_embeddings = Column(Text, nullable=True)  # For SQLite in tests
    else:
        delta_embeddings = Column(JSON, nullable=True)  # Store as JSON array of embeddings
    