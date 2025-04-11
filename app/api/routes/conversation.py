from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import asyncio

from app.db.database import get_db
from app.models.schemas import ConversationCreate, ConversationResponse
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.services.embedding_service import embedding_service

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"]
)

@router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(conversation_data: ConversationCreate, db: Session = Depends(get_db)):
    """
    Upload a conversation with messages and associate it with a user.
    If the user doesn't exist, it will be created.
    """
    # Check if user exists or create a new one
    user = db.query(User).filter(User.username == conversation_data.user_info.username).first()
    
    if not user:
        user = User(
            username=conversation_data.user_info.username,
            email=conversation_data.user_info.email,
            phone=conversation_data.user_info.phone,
            description=conversation_data.user_info.description
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # Create conversation
    new_conversation = Conversation(
        source=conversation_data.source,
        user_id=user.id
    )
    db.add(new_conversation)
    db.commit()
    db.refresh(new_conversation)
    
    # Process messages
    for msg_data in conversation_data.messages:
        # Find or create user for this message
        msg_user = db.query(User).filter(User.username == msg_data.user).first()
        if not msg_user:
            msg_user = User(username=msg_data.user)
            db.add(msg_user)
            db.commit()
            db.refresh(msg_user)
        
        # Create message without embedding initially
        new_message = Message(
            conversation_id=new_conversation.id,
            user_id=msg_user.id,
            content=msg_data.message,
            timestamp=msg_data.timestamp,
        )
        db.add(new_message)
        db.commit()
        db.refresh(new_message)
        
        # Schedule embedding generation with pgAI
        embedding_service.schedule_embedding_generation(
            db=db,
            text=msg_data.message,
            table="messages",
            id_column="id",
            id_value=new_message.id,
            text_column="content",
            embedding_column="embedding"
        )
    
    db.refresh(new_conversation)
    
    return new_conversation

@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Get a specific conversation by ID with all its messages."""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Conversation with ID {conversation_id} not found"
        )
    
    return conversation

@router.get("/", response_model=List[ConversationResponse])
def get_all_conversations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all conversations with pagination."""
    conversations = db.query(Conversation).offset(skip).limit(limit).all()
    return conversations 