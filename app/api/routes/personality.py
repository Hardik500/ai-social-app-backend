from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Union
import os
import httpx
import numpy as np
from pydantic import BaseModel

from app.db.database import get_db
from app.models.user import User
from app.models.personality import PersonalityProfile
from app.models.schemas import PersonalityProfileResponse, QuestionRequest, AnswerResponse, UserResponse
from app.services.personality_service import personality_service
from app.services.embedding_service import embedding_service
from app.models.conversation import Message
from app.services.response_validator import response_validator

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

class BulkEmailRequest(BaseModel):
    emails: List[str]

@router.post("/emails/bulk-generate", response_model=Dict[str, Any])
async def bulk_generate_personality_profiles(
    request: BulkEmailRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Generate personality profiles for multiple users based on their emails.
    
    This endpoint takes a list of email addresses and starts background tasks
    to generate personality profiles for each valid user.
    
    The profile generation happens in the background to prevent blocking the API.
    """
    results = {
        "total": len(request.emails),
        "successful": 0,
        "failed": 0,
        "details": []
    }
    
    for email in request.emails:
        # Find the user by email
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            results["failed"] += 1
            results["details"].append({
                "email": email,
                "status": "failed",
                "reason": f"User with email '{email}' not found"
            })
            continue
        
        # Verify the user has enough messages to generate a profile
        message_count = db.query(Message).filter(Message.user_id == user.id).count()
        if message_count < 5:  # Require at least 5 messages for a good profile
            results["failed"] += 1
            results["details"].append({
                "email": email,
                "status": "failed",
                "reason": f"Insufficient messages. User needs at least 5 messages."
            })
            continue
        
        # Start background task for profile generation
        background_tasks.add_task(
            personality_service.generate_profile,
            user.id,
            db
        )
        
        results["successful"] += 1
        results["details"].append({
            "email": email,
            "status": "processing",
            "user_id": user.id,
            "username": user.username
        })
    
    return results

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

# Helper functions for question handling
async def _handle_question_request(
    user: User,
    question: QuestionRequest,
    stream: bool, 
    db: Session
) -> Union[StreamingResponse, AnswerResponse]:
    """
    Common handler for processing question requests across different routes.
    
    This function handles the core business logic for processing questions:
    - Checking for active profile
    - Handling streaming vs non-streaming
    - Generating responses
    - Creating conversation context
    - Building response object
    
    Parameters:
    - user: The User object that the question is directed to
    - question: The question being asked
    - stream: Whether to stream the response
    - db: Database session
    """
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
        # Pass multi_message flag if present
        use_multi_message = getattr(question, 'multi_message', False)
        return StreamingResponse(
            personality_service.generate_response_stream(
                user.id, 
                question.question, 
                db,
                multi_message=use_multi_message
            ),
            media_type="application/json"
        )
    
    # Generate response (use cached response if available)
    print(f"Generating response for question: {question.question[:50]}...")
    
    # Pass multi_message flag if present
    use_multi_message = getattr(question, 'multi_message', False)
    answers = await personality_service.generate_response(
        user.id, 
        question.question, 
        db, 
        log_history=True,
        multi_message=use_multi_message
    )
    
    if not answers:
        print(f"Failed to generate response for user {user.username}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response"
        )
    
    print(f"Successfully generated response for user {user.username}, answer: {answers}")

    # Extract previous exchanges (last 5 conversation history entries for this user)
    previous_exchanges = []
    history_entries = db.query(Message).filter(Message.user_id == user.id).order_by(Message.created_at.desc()).limit(5).all()
    for entry in reversed(history_entries):
        previous_exchanges.append({
            "role": "user" if entry.user_id == user.id else "ai",
            "content": entry.content,
            "timestamp": str(entry.created_at)
        })

    # Extract preferred communication style if available
    preferred_communication_style = None
    if profile.traits and isinstance(profile.traits, dict):
        comm_style = profile.traits.get("communication_style")
        if isinstance(comm_style, str):
            preferred_communication_style = comm_style
        elif isinstance(comm_style, dict):
            # Join dict values for a summary
            preferred_communication_style = ", ".join(f"{k}: {v}" for k, v in comm_style.items())
    if not preferred_communication_style and hasattr(profile, "description"):
        preferred_communication_style = profile.description

    print(f"Validating response for user {user.username}, answer: {answers}")
    validation_result = await response_validator.validate_response(
        question=question.question,
        response=answers[0]["content"],
        personality_context={
            "traits": profile.traits,
            "communication_style": profile.traits.get("communication_style", {}),
            "interests": profile.traits.get("interests", [])
        },
        previous_exchanges=previous_exchanges,
        preferred_communication_style=preferred_communication_style
    )
    print(f"Validation result for user {user.username}, answer: {answers}: {validation_result}")

    # If the answer is incomplete/invalid, regenerate once
    if not validation_result.is_valid:
        print(f"Regenerating answer for user {user.username} due to validation failure, answer: {answers}...")
        answers = await personality_service.generate_response(
            user.id,
            question.question,
            db,
            log_history=True,
            multi_message=use_multi_message
        )
        if not answers:
            print(f"Failed to regenerate response for user {user.username}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate valid response after retry"
            )
        print(f"Re-validating regenerated response for user {user.username}, answer: {answers}...")
        validation_result = await response_validator.validate_response(
            question=question.question,
            response=answers[0]["content"],
            personality_context={
                "traits": profile.traits,
                "communication_style": profile.traits.get("communication_style", {}),
                "interests": profile.traits.get("interests", [])
            },
            previous_exchanges=previous_exchanges,
            preferred_communication_style=preferred_communication_style
        )
        print(f"Validation result after regeneration for user {user.username}, answer: {answers}: {validation_result}")

    print(f"Generating follow-up for user {user.username}, answer: {answers}...")
    followup = None
    if validation_result.is_valid and validation_result.needs_followup:
        followup = await response_validator.generate_followup(
            question=question.question,
            response=answers[0]["content"],
            personality_traits=profile.traits,
            engagement_level=validation_result.engagement_level,
            previous_exchanges=previous_exchanges,
            preferred_communication_style=preferred_communication_style
        )
        if followup and followup.get("followup_question"):
            answers.append({
                "content": followup["followup_question"],
                "type": "followup",
                "reasoning": followup.get("reasoning")
            })
    print(f"Followup generated for user {user.username}: {followup}")

    # Create conversation context with relevant information
    conversation_context = {
        "topic": _detect_topic(question.question),
        "tone": _detect_tone(question.question),
        "interests_matched": _find_matching_interests(question.question, profile.traits.get("interests", [])),
        "validation": {
            "score": validation_result.score,
            "engagement_level": validation_result.engagement_level,
            "issues": validation_result.issues
        }
    }
    
    return AnswerResponse(
        question=question.question,
        answers=answers,
        username=user.username,
        conversation_context=conversation_context
    )

def _get_user_by_username(username: str, db: Session) -> User:
    """Get a user by username with appropriate error handling."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with username '{username}' not found"
        )
    return user

def _get_user_by_email(email: str, db: Session) -> User:
    """Get a user by email with appropriate error handling."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found"
        )
    return user

@router.post("/users/{username}/ask", response_model=AnswerResponse)
async def ask_question(
    username: str, 
    question: QuestionRequest, 
    stream: bool = False,
    db: Session = Depends(get_db)
):
    """
    Ask a question to a user and get a response based on their personality profile.
    
    This endpoint uses the user's active personality profile to generate a response
    that matches their communication style and personality traits. The response may
    contain multiple messages to create more engaging and meaningful conversations.
    
    Parameters:
    - username: Username of the user to ask
    - question: Question to ask the user
    - stream: If true, return a streaming response (helpful for longer responses)
    """
    # Find the user
    user = _get_user_by_username(username, db)
    
    # Handle the question with common logic
    return await _handle_question_request(user, question, stream, db)

@router.post("/email/{email}/ask", response_model=AnswerResponse)
async def ask_question_by_email(
    email: str, 
    question: QuestionRequest, 
    stream: bool = False,
    db: Session = Depends(get_db)
):
    """
    Ask a question to a user by email and get a response based on their personality profile.
    
    This endpoint uses the user's active personality profile to generate a response
    that matches their communication style and personality traits. The response may
    contain multiple messages to create more engaging and meaningful conversations.
    
    Parameters:
    - email: Email of the user to ask
    - question: Question to ask the user
    - stream: If true, return a streaming response (helpful for longer responses)
    """
    # Find the user by email
    user = _get_user_by_email(email, db)
    
    # Handle the question with common logic
    return await _handle_question_request(user, question, stream, db)

@router.post("/users/{username}/ask/rag")
async def ask_personality_with_rag(
    username: str, 
    question: QuestionRequest,
    max_context_messages: int = 5,
    model: str = None,
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

# Helper functions for enhanced conversation context
def _detect_topic(question: str) -> str:
    """Detect the general topic of a question."""
    question_lower = question.lower()
    if any(word in question_lower for word in ["work", "job", "career", "project", "task"]):
        return "work"
    elif any(word in question_lower for word in ["family", "friend", "relationship", "partner"]):
        return "relationships"
    elif any(word in question_lower for word in ["hobby", "interest", "fun", "enjoy"]):
        return "interests"
    elif any(word in question_lower for word in ["health", "fitness", "exercise", "diet"]):
        return "health"
    elif any(word in question_lower for word in ["tech", "technology", "computer", "app", "software"]):
        return "technology"
    return "general"

def _detect_tone(question: str) -> str:
    """Detect the tone of a question."""
    question_lower = question.lower()
    if any(word in question_lower for word in ["urgent", "emergency", "critical", "asap"]):
        return "urgent"
    elif any(word in question_lower for word in ["sad", "sorry", "upset", "disappointed"]):
        return "empathetic"
    elif any(word in question_lower for word in ["excited", "happy", "great", "awesome"]):
        return "positive"
    elif any(word in question_lower for word in ["confused", "understand", "unclear", "explain"]):
        return "explanatory"
    elif any(word in question_lower for word in ["agree", "disagree", "opinion", "think"]):
        return "thoughtful"
    return "neutral"

def _find_matching_interests(question: str, interests: List) -> List[str]:
    """Find interests from the user's profile that match the question."""
    if not interests:
        return []
    
    # Convert interests to list if it's a string
    if isinstance(interests, str):
        interests = [i.strip() for i in interests.split(',')]
    
    question_lower = question.lower()
    matching = []
    for interest in interests:
        if isinstance(interest, str) and interest.lower() in question_lower:
            matching.append(interest)
    
    return matching 