#!/usr/bin/env python3
"""
Test script for the optimized personality service
"""
import asyncio
import os
import json
import sys
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all models first to avoid circular dependencies
from app.models.user import User
from app.models.conversation import Message, Conversation
from app.models.personality import PersonalityProfile

# Then import the database and services
from app.db.database import SessionLocal
from app.services.personality_service import personality_service

async def test_personality_generation(email: str = None, username: str = None):
    """Test personality generation for a specific user"""
    # Create a database session
    db = SessionLocal()
    
    try:
        # Find user by email or username
        user = None
        if email:
            user = db.query(User).filter(User.email == email).first()
        elif username:
            user = db.query(User).filter(User.username == username).first()
        
        if not user:
            if email:
                print(f"❌ User with email {email} not found")
            elif username:
                print(f"❌ User with username {username} not found")
            
            # List available users
            users = db.query(User).limit(10).all()
            if users:
                print("\nAvailable users:")
                for u in users:
                    print(f"- Username: {u.username}, Email: {u.email or 'None'}")
            return
            
        print(f"✅ Found user: {user.username} (ID: {user.id})")
        
        # Get existing personality profile if any
        existing_profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user.id,
            PersonalityProfile.is_active == True
        ).first()
        
        if existing_profile:
            print(f"Found existing profile (ID: {existing_profile.id})")
            print(f"- Generated: {existing_profile.generated_at}")
            print(f"- Message Count: {existing_profile.message_count}")
            print(f"- Has last_message_id: {'Yes' if existing_profile.last_message_id else 'No'}")
            print(f"- Has change_log: {'Yes' if existing_profile.change_log else 'No'}")
        else:
            print("No existing profile found")
            
        # Count messages
        message_count = db.query(Message).filter(Message.user_id == user.id).count()
        print(f"Total messages: {message_count}")
        
        if existing_profile and existing_profile.last_message_id:
            new_message_count = db.query(Message).filter(
                Message.user_id == user.id,
                Message.id > existing_profile.last_message_id
            ).count()
            print(f"New messages since last profile: {new_message_count}")
            
        # Generate or update profile
        print("\nGenerating/updating personality profile...")
        profile = await personality_service.generate_profile(user.id, db)
        
        if not profile:
            print("❌ Failed to generate personality profile")
            return
            
        print("\n✅ Successfully generated/updated personality profile:")
        print(f"- Profile ID: {profile.id}")
        print(f"- Generated/Updated: {profile.generated_at}")
        print(f"- Message Count: {profile.message_count}")
        print(f"- Last Message ID: {profile.last_message_id}")
        
        # Print traits summary
        print("\nPersonality Traits:")
        traits = profile.traits.get("traits", {})
        for trait, value in traits.items():
            print(f"- {trait.capitalize()}: {value}")
            
        # Print interests
        interests = profile.traits.get("interests", [])
        if interests:
            if isinstance(interests, list):
                print(f"\nInterests: {', '.join(interests)}")
            else:
                print(f"\nInterests: {interests}")
                
        # Print change log if present
        if hasattr(profile, 'change_log') and profile.change_log:
            print("\nChange Log:")
            if isinstance(profile.change_log, str):
                try:
                    change_log = json.loads(profile.change_log)
                    for entry in change_log:
                        print(f"- {entry.get('timestamp')}: Added {entry.get('new_message_count')} messages")
                        if 'changes' in entry:
                            print(f"  Changes: {entry['changes']}")
                except:
                    print(f"Change log (raw): {profile.change_log}")
            else:
                print(f"Change log: {profile.change_log}")
                
    finally:
        db.close()

if __name__ == "__main__":
    import sys
    
    email = None
    username = None
    
    if len(sys.argv) > 1:
        if '@' in sys.argv[1]:
            email = sys.argv[1]
            print(f"Testing personality generation for email: {email}")
        else:
            username = sys.argv[1]
            print(f"Testing personality generation for username: {username}")
    else:
        print("No user specified. Will list available users.")
        
    asyncio.run(test_personality_generation(email, username)) 