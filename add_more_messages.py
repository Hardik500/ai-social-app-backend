#!/usr/bin/env python3
"""
Script to add even more messages for testing incremental personality updates
"""
import asyncio
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all models to avoid circular dependencies
from app.models.user import User
from app.models.conversation import Message, Conversation
from app.models.personality import PersonalityProfile
from app.db.database import SessionLocal, engine, Base

# Sample new messages with additional interests and traits
NEW_MESSAGES = [
    # Travel and adventure interest
    "I just booked a trip to Japan to experience their tech culture firsthand.",
    "Working remotely from different countries has broadened my perspective.",
    "I find that traveling to new places sparks my creativity in unexpected ways.",
    "After hiking to Machu Picchu, I was inspired to simplify our architecture.",
    "I think we should organize a company retreat for team bonding.",
    
    # Environmental consciousness
    "We should make our software more energy-efficient to reduce carbon footprint.",
    "I switched to an electric car last month and it's been a great experience.",
    "Let's implement a paperless documentation system for sustainability.",
    "I volunteer for local forest restoration projects on weekends.",
    "Sustainable tech practices should be part of our development philosophy.",
    
    # Leadership qualities
    "I've been mentoring three junior developers this quarter with great results.",
    "Our last sprint succeeded because we clearly defined goals and responsibilities.",
    "I proposed a new project management approach that was adopted company-wide.",
    "Taking initiative on the database refactoring helped us meet our deadline.",
    "I believe in leading by example rather than micromanaging the team."
]

def add_more_messages():
    """Add even more messages to test incremental personality updates"""
    db = SessionLocal()
    
    try:
        # Get the test user
        test_user = db.query(User).filter(User.username == "test_user").first()
        
        if not test_user:
            print("Error: Test user not found")
            return
            
        # Get the user's active conversation
        conversation = db.query(Conversation).filter(
            Conversation.user_id == test_user.id
        ).first()
        
        if not conversation:
            print("Error: Conversation not found")
            return
            
        # Count existing messages
        existing_messages = db.query(Message).filter(
            Message.user_id == test_user.id
        ).count()
        
        print(f"User has {existing_messages} existing messages")
        
        # Get current active personality profile
        profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == test_user.id,
            PersonalityProfile.is_active == True
        ).first()
        
        if profile:
            print(f"Found active personality profile (ID: {profile.id})")
            print(f"- Last message ID: {profile.last_message_id}")
            print(f"- Total message count: {profile.message_count}")
        else:
            print("No active personality profile found")
            
        # Add new messages
        message_count = len(NEW_MESSAGES)
        print(f"Adding {message_count} new messages...")
        
        # Calculate a starting date (right after the latest message)
        now = datetime.now()
        start_date = now - timedelta(hours=6)  # Start from 6 hours ago
            
        # Create messages with timestamps spread over the last few hours
        for i, content in enumerate(NEW_MESSAGES):
            # Calculate a timestamp
            message_date = start_date + timedelta(
                minutes=random.randint(0, 6 * 60)  # Random time in the last 6 hours
            )
            
            # Create the message
            message = Message(
                user_id=test_user.id,
                conversation_id=conversation.id,
                content=content,
                created_at=message_date
            )
            db.add(message)
            
            # Commit in batches
            if i % 5 == 0 or i == message_count - 1:
                db.commit()
                print(f"Added {min(i+1, message_count)} messages...")
        
        print(f"Successfully added {message_count} new messages")
        
        # Get the latest message for reference
        latest_message = db.query(Message).filter(
            Message.user_id == test_user.id
        ).order_by(Message.id.desc()).first()
        
        if latest_message:
            print(f"Latest message ID: {latest_message.id}")
            
    finally:
        db.close()

if __name__ == "__main__":
    add_more_messages()
    print("\nNew messages added successfully!")
    print("Run this to test personality update:")
    print("python test_personality.py test_user") 