from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio
import json

from app.db.database import get_db
from app.models.user import User
from app.models.personality import PersonalityProfile
from app.models.schemas import PersonalityProfileResponse, QuestionRequest, AnswerResponse
from app.services.personality_service import personality_service

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

@router.post("/users/{username}/generate", response_model=PersonalityProfileResponse)
async def generate_personality_profile(username: str, db: Session = Depends(get_db)):
    """
    Generate a personality profile for a user based on their message history.
    
    This endpoint analyzes a user's messages to create a detailed personality profile
    that can be used to simulate their communication style and responses.
    """
    # Find the user
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with username '{username}' not found"
        )
    
    # Generate profile
    profile = await personality_service.generate_profile(user.id, db)
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not generate profile. User needs at least 5 messages."
        )
    
    # Preload responses to common questions in the background
    asyncio.create_task(personality_service.preload_common_questions(
        user.id, COMMON_QUESTIONS, db
    ))
    
    return profile

@router.post("/email/{email}/generate", response_model=PersonalityProfileResponse)
async def generate_personality_profile_by_email(email: str, db: Session = Depends(get_db)):
    """
    Generate a personality profile for a user based on their email and message history.
    
    This endpoint uses email as a unique identifier to find the user, then analyzes
    their messages to create a detailed personality profile that can be used to
    simulate their communication style and responses.
    """
    # Find the user by email
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found"
        )
    
    # Generate profile
    profile = await personality_service.generate_profile(user.id, db)
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not generate profile. User needs at least 5 messages."
        )
    
    # Preload responses to common questions in the background
    asyncio.create_task(personality_service.preload_common_questions(
        user.id, COMMON_QUESTIONS, db
    ))
    
    return profile

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
    # Find the user by email
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found"
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
        username=user.username  # Use username from the user object
    ) 