from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Set
import asyncio
from sqlalchemy import text
import numpy as np
import statistics
from sqlalchemy import and_, or_

from app.db.database import get_db
from app.models.schemas import ConversationCreate, ConversationResponse
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.services.embedding_service import embedding_service

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"]
)

def is_potential_duplicate(db: Session, conversation_data: ConversationCreate, timestamp_tolerance: float = 5.0) -> Dict:
    """
    Check if a conversation is potentially a duplicate by comparing source and timestamps.
    
    Args:
        db: Database session
        conversation_data: The conversation data to check
        timestamp_tolerance: Time difference in seconds to consider timestamps as matching
        
    Returns:
        Dict with 'is_duplicate' bool and 'existing_conversation' if a duplicate is found
    """
    # Get the source
    source = conversation_data.source
    
    # Extract timestamps from messages and convert to float
    message_timestamps = [float(msg.timestamp.split('.')[0]) for msg in conversation_data.messages]
    
    if not message_timestamps:
        return {"is_duplicate": False, "existing_conversation": None}
    
    # Calculate median timestamp to use as reference point
    median_timestamp = statistics.median(message_timestamps)
    
    # Find min and max timestamp with tolerance
    min_timestamp = min(message_timestamps) - timestamp_tolerance
    max_timestamp = max(message_timestamps) + timestamp_tolerance
    
    # Query for conversations with the same source
    potential_duplicates = db.query(Conversation).filter(
        Conversation.source == source
    ).all()
    
    for conv in potential_duplicates:
        # Get all messages for this conversation
        messages = db.query(Message).filter(Message.conversation_id == conv.id).all()
        
        if not messages:
            continue
            
        # Extract timestamps from existing messages
        existing_timestamps = [float(msg.timestamp.split('.')[0]) for msg in messages]
        
        # Check if the timestamps overlap
        existing_min = min(existing_timestamps)
        existing_max = max(existing_timestamps)
        
        # Check if there's significant overlap between timestamp ranges
        if (min_timestamp <= existing_max and max_timestamp >= existing_min) or \
           (existing_min <= max_timestamp and existing_max >= min_timestamp):
            
            # Calculate how many messages have matching timestamps (within tolerance)
            matching_count = 0
            total_count = len(message_timestamps)
            
            for ts in message_timestamps:
                if any(abs(ts - existing_ts) <= timestamp_tolerance for existing_ts in existing_timestamps):
                    matching_count += 1
            
            # If more than 50% of messages match by timestamp, consider it a duplicate
            if matching_count / total_count > 0.5:
                return {
                    "is_duplicate": True, 
                    "existing_conversation": conv,
                    "match_percentage": matching_count / total_count * 100
                }
    
    return {"is_duplicate": False, "existing_conversation": None}

@router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(conversation_data: ConversationCreate, skip_duplicate_check: bool = False, db: Session = Depends(get_db)):
    """
    Upload a conversation with messages and associate it with a user.
    If the user doesn't exist, it will be created.
    Checks for duplicate conversations based on source and timestamps.
    
    Parameters:
    - conversation_data: The conversation data to upload
    - skip_duplicate_check: If True, bypass duplicate checking (default: False)
    """
    # Check for duplicates if not explicitly skipped
    if not skip_duplicate_check:
        duplicate_check = is_potential_duplicate(db, conversation_data)
        if duplicate_check["is_duplicate"]:
            # Return the existing conversation with a different status code to indicate it's a duplicate
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "id": duplicate_check["existing_conversation"].id,
                    "source": duplicate_check["existing_conversation"].source,
                    "user_id": duplicate_check["existing_conversation"].user_id,
                    "created_at": duplicate_check["existing_conversation"].created_at.isoformat(),
                    "messages": [
                        {
                            "id": msg.id,
                            "content": msg.content,
                            "timestamp": msg.timestamp,
                            "user_id": msg.user_id,
                            "conversation_id": msg.conversation_id,
                            "created_at": msg.created_at.isoformat()
                        } 
                        for msg in duplicate_check["existing_conversation"].messages
                    ],
                    "duplicate_detection": {
                        "is_duplicate": True,
                        "match_percentage": duplicate_check["match_percentage"]
                    }
                }
            )
    
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