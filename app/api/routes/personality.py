from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import asyncio
import json
import os
import httpx
import numpy as np

from app.db.database import get_db
from app.models.user import User
from app.models.personality import PersonalityProfile
from app.models.schemas import PersonalityProfileResponse, QuestionRequest, AnswerResponse, UserResponse
from app.services.personality_service import personality_service
from app.services.embedding_service import embedding_service
from app.models.conversation import Message

router = APIRouter(
    prefix="/personalities",
    tags=["personalities"]
)

# Common questions to preload for faster response times
COMMON_QUESTIONS = [
    "What do you think about this?",
    "How would you solve this problem?",
    "What's your opinion on this topic?",
    "Do you agree with this approach?",
    "Can you help me with this issue?"
]

@router.post("/users/{username}/generate", response_model=Dict[str, Any])
async def generate_personality_profile(
    username: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Generate a personality profile for a user based on their message history.
    
    This endpoint analyzes a user's messages to create a detailed personality profile
    that can be used to simulate their communication style and responses.
    
    The profile generation happens in the background to prevent blocking the API.
    """
    # Find the user
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with username '{username}' not found"
        )
    
    # Verify the user has enough messages to generate a profile
    message_count = db.query(Message).filter(Message.user_id == user.id).count()
    if message_count < 5:  # Require at least 5 messages for a good profile
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not generate profile. User needs at least 5 messages."
        )
    
    # Start background task for profile generation
    background_tasks.add_task(
        personality_service.generate_profile,
        user.id,
        db
    )
    
    # Return immediate response while generation happens in background
    return {
        "status": "processing",
        "message": f"Profile generation for {username} has started. Check back in a few moments.",
        "user_id": user.id,
        "username": username
    }

@router.post("/email/{email}/generate", response_model=Dict[str, Any])
async def generate_personality_profile_by_email(
    email: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Generate a personality profile for a user based on their email and message history.
    
    This endpoint uses email as a unique identifier to find the user, then analyzes
    their messages to create a detailed personality profile that can be used to
    simulate their communication style and responses.
    
    The profile generation happens in the background to prevent blocking the API.
    """
    # Find the user by email
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found"
        )
    
    # Verify the user has enough messages to generate a profile
    message_count = db.query(Message).filter(Message.user_id == user.id).count()
    if message_count < 5:  # Require at least 5 messages for a good profile
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not generate profile. User needs at least 5 messages."
        )
    
    # Start background task for profile generation
    background_tasks.add_task(
        personality_service.generate_profile,
        user.id,
        db
    )
    
    # Return immediate response while generation happens in background
    return {
        "status": "processing",
        "message": f"Profile generation for {user.username} (email: {email}) has started. Check back in a few moments.",
        "user_id": user.id,
        "username": user.username
    }

@router.get("/users/{username}", response_model=List[PersonalityProfileResponse])
def get_user_personality_profiles(username: str, active_only: bool = False, db: Session = Depends(get_db)):
    """
    Get all personality profiles for a user.
    
    Parameters:
    - username: Username of the user
    - active_only: If true, return only the active profile
    """
    # Find the user
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with username '{username}' not found"
        )
    
    # Query for profiles
    query = db.query(PersonalityProfile).filter(PersonalityProfile.user_id == user.id)
    
    if active_only:
        query = query.filter(PersonalityProfile.is_active == True)
    
    profiles = query.all()
    
    return profiles

@router.get("/email/{email}", response_model=List[PersonalityProfileResponse])
def get_user_personality_profiles_by_email(email: str, active_only: bool = False, db: Session = Depends(get_db)):
    """
    Get all personality profiles for a user identified by email.
    
    Parameters:
    - email: Email address of the user (unique identifier)
    - active_only: If true, return only the active profile
    """
    # Find the user by email
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found"
        )
    
    # Query for profiles
    query = db.query(PersonalityProfile).filter(PersonalityProfile.user_id == user.id)
    
    if active_only:
        query = query.filter(PersonalityProfile.is_active == True)
    
    profiles = query.all()
    
    return profiles

@router.get("/profiles/{profile_id}", response_model=PersonalityProfileResponse)
def get_profile_by_id(profile_id: int, db: Session = Depends(get_db)):
    """Get a specific personality profile by ID."""
    profile = db.query(PersonalityProfile).filter(PersonalityProfile.id == profile_id).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Personality profile with ID {profile_id} not found"
        )
    
    return profile

@router.post("/users/{username}/ask", response_model=AnswerResponse)
async def ask_question(
    username: str, 
    question: QuestionRequest, 
    stream: bool = False,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Ask a question to a user and get a response based on their personality profile.
    
    This endpoint uses the user's active personality profile to generate a response
    that matches their communication style and personality traits.
    
    Parameters:
    - username: Username of the user to ask
    - question: Question to ask the user
    - stream: If true, return a streaming response (helpful for longer responses)
    """
    # Find the user
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with username '{username}' not found"
        )
    
    # Check if user has an active profile
    profile = db.query(PersonalityProfile).filter(
        PersonalityProfile.user_id == user.id,
        PersonalityProfile.is_active == True
    ).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User doesn't have an active personality profile. Generate one first."
        )
    
    # If streaming is requested, use streaming response
    if stream:
        return StreamingResponse(
            personality_service.generate_response_stream(user.id, question.question, db),
            media_type="application/json"
        )
    
    # Generate response (use cached response if available)
    answer = await personality_service.generate_response(user.id, question.question, db)
    
    if not answer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response"
        )
    
    # Preload more responses in the background after successful response
    background_tasks.add_task(
        personality_service.preload_related_questions,
        user.id, question.question, answer, db
    )
    
    return AnswerResponse(
        question=question.question,
        answer=answer,
        username=username
    )
@router.post("/email/{email}/ask", response_model=AnswerResponse)
async def ask_question_by_email(
    email: str, 
    question: QuestionRequest, 
    stream: bool = False,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Ask a question to a user identified by email and get a response based on their personality profile.
    
    This endpoint finds the user by their unique email address, then uses their active
    personality profile to generate a response that matches their communication style and personality traits.
    
    Parameters:
    - email: Email of the user to ask
    - question: Question to ask the user
    - stream: If true, return a streaming response (helpful for longer responses)
    """
    print(f"Processing question request for email: {email}")
    
    # Find the user by email
    user = db.query(User).filter(User.email == email).first()
    if not user:
        print(f"User not found with email: {email}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found"
        )
    
    print(f"Found user: {user.username} (ID: {user.id})")
    
    # Check if user has an active profile
    profile = db.query(PersonalityProfile).filter(
        PersonalityProfile.user_id == user.id,
        PersonalityProfile.is_active == True
    ).first()
    
    if not profile:
        print(f"No active profile found for user {user.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User doesn't have an active personality profile. Generate one first."
        )
    
    print(f"Found active profile for user {user.username}")
    
    # If streaming is requested, use streaming response
    if stream:
        print(f"Streaming response requested for user {user.username}")
        return StreamingResponse(
            personality_service.generate_response_stream(user.id, question.question, db),
            media_type="application/json"
        )
    
    # Generate response (use cached response if available)
    print(f"Generating response for question: {question.question[:50]}...")
    answer = await personality_service.generate_response(user.id, question.question, db)
    
    if not answer:
        print(f"Failed to generate response for user {user.username}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response"
        )
    
    print(f"Successfully generated response for user {user.username}")
    
    # Preload more responses in the background after successful response
    print(f"Adding background task to preload related questions for user {user.username}")
    background_tasks.add_task(
        personality_service.preload_related_questions, 
        user.id, question.question, answer, db
    )
    
    return AnswerResponse(
        question=question.question,
        answer=answer,
        username=user.username  # Use username from the user object
    )

@router.post("/users/{username}/ask/rag")
async def ask_personality_with_rag(
    username: str, 
    question: QuestionRequest,
    max_context_messages: int = 5,
    model: str = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Ask a question to a user's personality, enhanced with RAG for more accurate responses.
    
    This endpoint:
    1. Finds relevant messages the user has actually sent (using vector similarity)
    2. Combines those messages with the user's personality profile
    3. Uses both to generate a more accurate, evidence-based response
    
    Parameters:
    - username: The username to generate a response for
    - question: The question to ask
    - max_context_messages: Maximum number of similar messages to include
    - model: Optional override for LLM model
    """
    # First, get the active personality profile
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found"
        )
    
    profile = db.query(PersonalityProfile).filter(
        PersonalityProfile.user_id == user.id,
        PersonalityProfile.is_active == True
    ).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active personality profile found for user '{username}'"
        )
    
    # Find semantically similar messages from this user
    relevant_messages = []
    try:
        # Generate embedding for the question
        question_embedding = await embedding_service.generate_embedding(question.question)
        
        # Find similar messages from this user using cosine similarity
        messages = db.query(Message).filter(Message.user_id == user.id).all()
        
        if messages:
            # Calculate similarity scores
            similarity_scores = []
            for msg in messages:
                if msg.embedding is not None:
                    # Calculate cosine similarity
                    dot_product = np.dot(np.array(msg.embedding), np.array(question_embedding))
                    norm_a = np.linalg.norm(np.array(msg.embedding))
                    norm_b = np.linalg.norm(np.array(question_embedding))
                    similarity = dot_product / (norm_a * norm_b)
                    
                    similarity_scores.append({
                        "message": msg.content,
                        "similarity": float(similarity)
                    })
            
            # Sort by similarity (highest first) and take top results
            similarity_scores.sort(key=lambda x: x["similarity"], reverse=True)
            relevant_messages = similarity_scores[:max_context_messages]
    except Exception as e:
        print(f"Error finding similar messages: {str(e)}")
        # Continue with no similar messages if there's an error
    
    # Use the personality service to build an enhanced system prompt
    system_prompt = await personality_service.build_rag_enhanced_system_prompt(
        user_id=user.id,
        username=username,
        relevant_messages=relevant_messages,
        question=question.question,
        db=db
    )
    
    # Generate response using Ollama API
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = model or os.getenv("OLLAMA_CHAT_MODEL", "llama3")
    
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.question}
    ]
    
    # Call Ollama API
    try:
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
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error from Ollama API: {response.status_code} - {response.text}"
                )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )
    
    # Preload more responses in the background after successful response
    background_tasks.add_task(
        personality_service.preload_related_questions,
        user.id, question.question, answer, db
    )
    
    # Return the enhanced response with context information
    return {
        "question": question.question,
        "answer": answer,
        "username": username,
        "used_messages": len(relevant_messages),
        "relevant_context": [msg["message"] for msg in relevant_messages]
    }

@router.get("/active-users", response_model=List[UserResponse])
def get_active_users(db: Session = Depends(get_db)):
    """
    Get all users who have active personality profiles.
    
    This endpoint returns a list of users who have at least one active
    personality profile that can be used for chat interactions.
    
    If no users with active profiles are found, it falls back to returning all users.
    """
    # First try to get users with active personality profiles
    users_with_profiles = db.query(User).join(
        PersonalityProfile, 
        User.id == PersonalityProfile.user_id
    ).filter(
        PersonalityProfile.is_active == True
    ).distinct().all()
    
    # If we found users with active profiles, return them
    if users_with_profiles:
        return users_with_profiles
    
    # Otherwise, return all users as a fallback
    return db.query(User).all()

@router.get("/all-users", response_model=List[UserResponse])
def get_all_users(db: Session = Depends(get_db)):
    """
    Get all users regardless of personality profile status.
    """
    users = db.query(User).all()
    
    if not users:
        return []
    
    return users

@router.get("/users/{username}/profile-status", response_model=Dict[str, Any])
async def check_profile_generation_status(username: str, db: Session = Depends(get_db)):
    """
    Check the status of a user's personality profile generation.
    
    Returns information about whether a profile is currently being generated,
    or if one already exists.
    """
    # Find the user
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with username '{username}' not found"
        )
    
    # Check if there's an active profile
    profile = db.query(PersonalityProfile).filter(
        PersonalityProfile.user_id == user.id,
        PersonalityProfile.is_active == True
    ).first()
    
    if profile:
        return {
            "status": "completed",
            "message": "Profile generation is complete.",
            "profile_id": profile.id,
            "created_at": profile.created_at.isoformat() if hasattr(profile, 'created_at') else None,
            "user_id": user.id,
            "username": username
        }
    
    # No active profile found
    return {
        "status": "pending",
        "message": "No active profile found. If you requested profile generation, it may still be processing.",
        "user_id": user.id,
        "username": username
    }

@router.get("/email/{email}/profile-status", response_model=Dict[str, Any])
async def check_profile_generation_status_by_email(email: str, db: Session = Depends(get_db)):
    """
    Check the status of a user's personality profile generation by email.
    
    Returns information about whether a profile is currently being generated,
    or if one already exists.
    """
    # Find the user by email
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found"
        )
    
    # Check if there's an active profile
    profile = db.query(PersonalityProfile).filter(
        PersonalityProfile.user_id == user.id,
        PersonalityProfile.is_active == True
    ).first()
    
    if profile:
        return {
            "status": "completed",
            "message": "Profile generation is complete.",
            "profile_id": profile.id,
            "created_at": profile.created_at.isoformat() if hasattr(profile, 'created_at') else None,
            "user_id": user.id,
            "username": user.username
        }
    
    # No active profile found
    return {
        "status": "pending",
        "message": "No active profile found. If you requested profile generation, it may still be processing.",
        "user_id": user.id,
        "username": user.username
    } 