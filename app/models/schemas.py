from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None
    description: Optional[str] = None

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Message schemas
class MessageBase(BaseModel):
    user: str
    timestamp: str
    message: str

class MessageCreate(MessageBase):
    pass

class MessageResponse(BaseModel):
    id: int
    content: str
    timestamp: str
    user_id: int
    conversation_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Conversation schemas
class ConversationCreate(BaseModel):
    source: Optional[str] = "slack"
    messages: List[MessageBase]
    primary_user_info: UserBase
    additional_users: List[UserBase] = []

class ConversationResponse(BaseModel):
    id: int
    source: str
    user_id: int
    created_at: datetime
    messages: List[MessageResponse]
    
    class Config:
        from_attributes = True

# Personality Profile schemas
class PersonalityProfileBase(BaseModel):
    traits: Dict[str, Any]
    description: str
    system_prompt: str

class PersonalityProfileCreate(PersonalityProfileBase):
    pass

class PersonalityProfileResponse(PersonalityProfileBase):
    id: int
    user_id: int
    generated_at: datetime
    updated_at: Optional[datetime]
    is_active: bool
    message_count: int
    
    class Config:
        from_attributes = True

# Question/Answer schemas
class QuestionRequest(BaseModel):
    question: str

class MessageContent(BaseModel):
    content: str
    type: str = "text"  # Options: text, thinking, media, etc.

class AnswerResponse(BaseModel):
    question: str
    answers: List[MessageContent]  # Multiple messages in the response
    username: str
    conversation_context: Optional[Dict[str, Any]] = None  # Optional context about the conversation

class UserInfoSchema(BaseModel):
    """Schema for user information."""
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None
    description: Optional[str] = None

class AdditionalUserSchema(BaseModel):
    """Schema for additional user information."""
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None
    description: Optional[str] = None

class IngestionResultSchema(BaseModel):
    """Schema for ingestion results."""
    status: str
    conversation_id: Optional[int] = None
    messages_imported: Optional[int] = None
    message: Optional[str] = None

class ConversationHistoryCreate(BaseModel):
    user_id: int
    role: str  # 'user' or 'ai'
    content: str

class ConversationHistoryResponse(BaseModel):
    id: int
    user_id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True 