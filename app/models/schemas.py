from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
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