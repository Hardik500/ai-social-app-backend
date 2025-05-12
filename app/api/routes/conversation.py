from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Set
import asyncio
from sqlalchemy import text
import numpy as np
import statistics
from sqlalchemy import and_, or_
from datetime import datetime
import os
import httpx

from app.db.database import get_db
from app.models.schemas import ConversationCreate, ConversationResponse, ConversationHistoryCreate, ConversationHistoryResponse
from app.models.user import User
from app.models.conversation import Conversation, Message, ConversationHistory
from app.services.embedding_service import embedding_service, embedding_auto_updater

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"]
)

def is_potential_duplicate(db: Session, conversation_data: ConversationCreate, timestamp_tolerance: float = 5.0) -> Dict:
    """
    Check if a conversation is a potential duplicate by comparing timestamps and source.
    
    Args:
        db: Database session
        conversation_data: The new conversation data to check
        timestamp_tolerance: Time difference threshold in seconds
        
    Returns:
        Dict with 'is_duplicate' bool and 'existing_conversation' if a duplicate is found
    """
    # Get the source
    source = conversation_data.source
    
    # Parse ISO timestamps and convert to UNIX timestamps
    message_timestamps = []
    for msg in conversation_data.messages:
        try:
            # Parse ISO format timestamp to datetime
            dt = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))
            # Convert to UNIX timestamp (seconds since epoch)
            message_timestamps.append(dt.timestamp())
        except ValueError:
            # If timestamp can't be parsed, use current time
            message_timestamps.append(datetime.now().timestamp())
    
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
        existing_timestamps = []
        for msg in messages:
            try:
                # Parse ISO format timestamp to datetime
                dt = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))
                # Convert to UNIX timestamp
                existing_timestamps.append(dt.timestamp())
            except ValueError:
                # If timestamp can't be parsed, skip this message
                continue
        
        if not existing_timestamps:
            continue
        
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
    user = db.query(User).filter(User.username == conversation_data.primary_user_info.username).first()
    
    if not user:
        user = User(
            username=conversation_data.primary_user_info.username,
            email=conversation_data.primary_user_info.email,
            phone=conversation_data.primary_user_info.phone,
            description=conversation_data.primary_user_info.description
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
   
    # Process additional users if provided
    for additional_user_info in conversation_data.additional_users:
        additional_user = db.query(User).filter(User.username == additional_user_info.username).first()
        if not additional_user:
            additional_user = User(
                username=additional_user_info.username,
                email=additional_user_info.email,
                phone=additional_user_info.phone,
                description=additional_user_info.description
            )
            db.add(additional_user)
            db.commit()
            db.refresh(additional_user)
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
        # Find user for this message (might already exist from additional_users)
        msg_user = db.query(User).filter(User.username == msg_data.user).first()
        if not msg_user:
            msg_user = User(username=msg_data.user)
            db.add(msg_user)
            db.commit()
            db.refresh(msg_user)
        
        # Create message - initially without embedding
        new_message = Message(
            conversation_id=new_conversation.id,
            user_id=msg_user.id,
            content=msg_data.message,
            timestamp=msg_data.timestamp
        )
        db.add(new_message)
        db.commit()
        db.refresh(new_message)
        
        # Schedule embedding generation in the background
        embedding_auto_updater.schedule_update(
            db=db,
            table="messages",
            id_column="id",
            id_value=new_message.id,
            text_column="content",
            embedding_column="embedding"
        )
    
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
async def search_similar_messages(query: str, limit: int = 5, db: Session = Depends(get_db)):
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
        query_embedding = await embedding_service.generate_embedding(query)
        
        # Calculate cosine similarity manually
        similar_messages = []
        
        # Function to calculate cosine similarity between two vectors
        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            return dot_product / (norm_a * norm_b)
        
        for message, username in messages:
            if message.embedding is None or len(message.embedding) == 0:
                continue
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

@router.post("/rag")
async def retrieval_augmented_generation(
    query: str, 
    max_context_messages: int = 5, 
    model: str = None,
    db: Session = Depends(get_db)
):
    """
    RAG endpoint that retrieves relevant messages as context and uses an LLM to generate an answer.
    
    - Retrieves semantically similar messages based on vector similarity
    - Uses these messages as context for the LLM to answer the query
    - Returns both the answer and the relevant context that was used
    
    Parameters:
    - query: The question to answer
    - max_context_messages: Maximum number of messages to use as context (default: 5)
    - model: Optional override for the LLM model to use
    
    Example:
    ```
    POST /conversations/rag?query=What is the project deadline?&max_context_messages=3
    
    Response:
    {
        "query": "What is the project deadline?",
        "answer": "Based on the provided context, the project deadline is March 15th, 2023.",
        "context_used": [
            "Message from john_doe: We need to finish the project by March 15th, 2023. (Similarity: 0.89)",
            "Message from project_manager: Don't forget about the deadline next month! (Similarity: 0.76)",
            "Message from team_lead: The client expects all deliverables by mid-March. (Similarity: 0.71)"
        ],
        "model_used": "llama3"
    }
    ```
    """
    try:
        # First, get similar messages using embeddings
        similar_results = await search_similar_messages(query, limit=max_context_messages, db=db)
        similar_messages = similar_results.get("similar_messages", [])
        
        if not similar_messages:
            return {
                "query": query,
                "answer": "I couldn't find any relevant information to answer this question.",
                "context_used": [],
                "error": "No similar messages found in the database"
            }
        
        # Prepare context from similar messages
        context_texts = []
        for msg in similar_messages:
            context_texts.append(f"Message from {msg['username']}: {msg['content']} (Similarity: {msg['similarity_score']:.2f})")
        
        context_text = "\n\n".join(context_texts)
        
        # Use Ollama API to generate answer
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = model or os.getenv("OLLAMA_CHAT_MODEL", "llama3")
        
        # Build the prompt with context
        system_prompt = "You are a helpful assistant. Answer the question based ONLY on the provided context. If the context doesn't contain the answer, say you don't have enough information."
        
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context information:\n\n{context_text}\n\nQuestion: {query}\n\nYour answer:"}
        ]
        
        # Call Ollama API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ollama_base_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": chat_messages,
                    "stream": False
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["message"]["content"]
            else:
                return {
                    "query": query,
                    "answer": None,
                    "context_used": context_texts,
                    "error": f"Error from Ollama API: {response.status_code} - {response.text}"
                }
        
        # Return both the answer and the context that was used
        return {
            "query": query,
            "answer": answer,
            "context_used": context_texts,
            "model_used": ollama_model
        }
        
    except Exception as e:
        return {
            "query": query,
            "answer": None,
            "error": str(e),
            "context_used": []
        } 

@router.get("/history/{user_id}", response_model=List[ConversationHistoryResponse])
def get_conversation_history(user_id: int, db: Session = Depends(get_db)):
    """Get all conversation history for a user (single session)."""
    return db.query(ConversationHistory).filter(ConversationHistory.user_id == user_id).order_by(ConversationHistory.created_at).all()

@router.get("/history/email/{email}", response_model=List[ConversationHistoryResponse])
def get_conversation_history_by_email(email: str, db: Session = Depends(get_db)):
    """Get all conversation history for a user by email (single session)."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found"
        )
    return db.query(ConversationHistory).filter(ConversationHistory.user_id == user.id).order_by(ConversationHistory.created_at).all()

@router.post("/history/", response_model=ConversationHistoryResponse)
def add_conversation_history(entry: ConversationHistoryCreate, db: Session = Depends(get_db)):
    """Add a message to the conversation history (single session)."""
    new_entry = ConversationHistory(**entry.dict())
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return new_entry

@router.post("/history/email/", response_model=ConversationHistoryResponse)
def add_conversation_history_by_email(
    entry: ConversationHistoryCreate, 
    email: str,
    db: Session = Depends(get_db)
):
    """Add a message to the conversation history using email instead of user_id."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found"
        )
    
    # Override the user_id with the one found from email
    entry_dict = entry.dict()
    entry_dict["user_id"] = user.id
    
    new_entry = ConversationHistory(**entry_dict)
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return new_entry

@router.delete("/history/{user_id}")
def clear_conversation_history(user_id: int, db: Session = Depends(get_db)):
    """Clear all conversation history for a user (single session)."""
    db.query(ConversationHistory).filter(ConversationHistory.user_id == user_id).delete()
    db.commit()
    return {"status": "success", "message": f"Cleared conversation history for user {user_id}"}

@router.delete("/history/email/{email}")
def clear_conversation_history_by_email(email: str, db: Session = Depends(get_db)):
    """Clear all conversation history for a user by email (single session)."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found"
        )
    db.query(ConversationHistory).filter(ConversationHistory.user_id == user.id).delete()
    db.commit()
    return {"status": "success", "message": f"Cleared conversation history for user with email {email}"}

@router.delete("/messages/user/{user_id}")
def delete_user_messages(user_id: int, db: Session = Depends(get_db)):
    """Delete all messages from a specific user by user ID."""
    # Check if user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    # Get all messages from the user
    messages = db.query(Message).filter(Message.user_id == user_id).all()
    
    # If no messages found, return appropriate response
    if not messages:
        return {
            "status": "success",
            "message": f"No messages found for user with ID {user_id}",
            "count": 0
        }
    
    # Count messages for reporting
    count = len(messages)
    
    # Delete all messages from the user
    db.query(Message).filter(Message.user_id == user_id).delete()
    db.commit()
    
    return {
        "status": "success",
        "message": f"Deleted {count} messages from user with ID {user_id}",
        "count": count
    }

@router.delete("/messages/email/{email}")
def delete_user_messages_by_email(email: str, db: Session = Depends(get_db)):
    """Delete all messages from a specific user by email."""
    # Check if user exists
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email {email} not found"
        )
    
    # Get all messages from the user
    messages = db.query(Message).filter(Message.user_id == user.id).all()
    
    # If no messages found, return appropriate response
    if not messages:
        return {
            "status": "success",
            "message": f"No messages found for user with email {email}",
            "count": 0
        }
    
    # Count messages for reporting
    count = len(messages)
    
    # Delete all messages from the user
    db.query(Message).filter(Message.user_id == user.id).delete()
    db.commit()
    
    return {
        "status": "success",
        "message": f"Deleted {count} messages from user with email {email}",
        "count": count
    } 