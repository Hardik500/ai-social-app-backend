from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import pgvector.sqlalchemy

from app.db.database import Base

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
    embedding = Column(pgvector.sqlalchemy.Vector(768))
    
    # Whether this is the active profile for the user (users can have multiple profiles)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="personality_profiles")
    
    # All messages used to generate this profile
    message_count = Column(Integer, default=0)
    
    # System prompt to use for personality simulation
    system_prompt = Column(Text) 