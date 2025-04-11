from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import asyncio
from sqlalchemy import text
import numpy as np

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
        
        # Generate embedding for message
        embedding = await embedding_service.generate_embedding(msg_data.message)
        
        # Create message with embedding
        new_message = Message(
            conversation_id=new_conversation.id,
            user_id=msg_user.id,
            content=msg_data.message,
            timestamp=msg_data.timestamp,
            embedding=embedding
        )
        db.add(new_message)
    
    db.commit()
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

@router.get("/messages/{message_id}/embedding")
def get_message_embedding(message_id: int, db: Session = Depends(get_db)):
    """Get a specific message's embedding to verify it exists."""
    message = db.query(Message).filter(Message.id == message_id).first()
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Message with ID {message_id} not found"
        )
    
    # Return message details with embedding
    return {
        "id": message.id,
        "content": message.content,
        "embedding_exists": message.embedding is not None,
        "embedding_dimensions": len(message.embedding) if message.embedding is not None else None,
        "embedding_sample": message.embedding[:5].tolist() if message.embedding is not None else None
    }

@router.get("/search/similar")
def search_similar_messages(query: str, limit: int = 5, db: Session = Depends(get_db)):
    """
    Search for messages similar to the query string.
    Uses cosine similarity to find semantically similar messages.
    """
    try:
        # First, we need to get all messages with their embeddings
        messages = db.query(Message, User.username).join(User).all()
        
        if not messages:
            return {
                "query": query,
                "similar_messages": [],
                "message": "No messages found to compare against"
            }
        
        # Generate embedding for the query
        query_embedding = asyncio.run(embedding_service.generate_embedding(query))
        
        # Calculate cosine similarity manually
        similar_messages = []
        
        # Function to calculate cosine similarity between two vectors
        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            return dot_product / (norm_a * norm_b)
        
        for message, username in messages:
            # Calculate similarity
            similarity = cosine_similarity(np.array(message.embedding), np.array(query_embedding))
            
            similar_messages.append({
                "id": message.id,
                "content": message.content,
                "username": username,
                "conversation_id": message.conversation_id,
                "similarity_score": float(similarity)
            })
        
        # Sort by similarity (highest first)
        similar_messages.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Return only the top 'limit' results
        return {
            "query": query,
            "similar_messages": similar_messages[:limit]
        }
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "similar_messages": []
        } 