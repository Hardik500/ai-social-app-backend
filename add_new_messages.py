#!/usr/bin/env python3
"""
Script to add new messages for testing incremental personality updates
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

# Sample new messages with different traits/interests/values
NEW_MESSAGES = [
    # More technical messages
    "I've been exploring quantum computing principles in my spare time.",
    "I think we should switch to GraphQL for our API to improve efficiency.",
    "I wrote a custom caching layer that reduced our database load by 40%.",
    "The new neural network architecture I implemented shows promising results.",
    "I believe blockchain technology has applications beyond cryptocurrency.",
    
    # More creativity-related
    "I've started watercolor painting on weekends to explore my creative side.",
    "I composed a piece of music inspired by our latest product launch.",
    "Let's think outside the box on this problem - what if we approached it from a different angle?",
    "I love experimenting with new design patterns even if they're unconventional.",
    "Art and technology can complement each other in surprising ways.",
    
    # More social/team-focused
    "I organized a team-building event for next month to strengthen our bonds.",
    "Communication is key - we should have more regular sync-ups with the design team.",
    "I prefer pair programming for complex tasks; two minds are better than one.",
    "The team's morale has improved since we implemented flexible working hours.",
    "I volunteer as a coding mentor for local students on weekends.",
    
    # More detail-oriented
    "I reviewed the entire codebase and found 17 unused functions we can remove.",
    "The UI has a 2-pixel misalignment on the dashboard when viewed at 125% zoom.",
    "Let me check each test case individually to ensure complete coverage.",
    "I noticed our error rate increases by 0.4% during peak hours - we should investigate.",
    "The documentation needs to be updated with the exact parameter definitions."
]

def add_new_messages():
    """Add new messages to test incremental personality updates"""
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
            
        # Add 20 new messages
        message_count = len(NEW_MESSAGES)
        print(f"Adding {message_count} new messages...")
        
        # Calculate a starting date (right after the latest message)
        now = datetime.now()
        start_date = now - timedelta(days=1)  # Start from yesterday
            
        # Create messages with timestamps spread over the last day
        for i, content in enumerate(NEW_MESSAGES):
            # Calculate a timestamp
            message_date = start_date + timedelta(
                minutes=random.randint(0, 24 * 60)  # Random time in the last 24 hours
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
    add_new_messages()
    print("\nNew messages added successfully!")
    print("Run this to test incremental personality update:")
    print("python test_personality.py test_user") 