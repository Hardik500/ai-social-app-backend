#!/usr/bin/env python3
"""
Script to create a test user and messages for personality testing
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

# Sample messages for our test user
SAMPLE_MESSAGES = [
    # Technical messages
    "I've been working on optimizing our database queries to improve performance.",
    "The new React component I built is really efficient with state management.",
    "I think we should use TypeScript for this project to catch type errors early.",
    "Docker containers make deployment so much easier across different environments.",
    "I'm really excited about the potential of AI in our product roadmap.",
    
    # Collaborative messages
    "Let's schedule a meeting to discuss this further with the team.",
    "I appreciate your feedback on my pull request, it helped improve the code.",
    "I think we should involve the UX team early in this project.",
    "I'm happy to help you with that task if you're overloaded.",
    "Great work on the presentation yesterday, the client was impressed!",
    
    # Problem-solving
    "I found a bug in the authentication flow, let me fix it before release.",
    "We could refactor this to use a more efficient algorithm.",
    "Let's break down this complex problem into smaller, manageable parts.",
    "I think we need to reconsider our approach to state management here.",
    "The root cause was a race condition in the async code execution.",
    
    # Thoughtful/reflective
    "I've been thinking about how we could improve our dev process.",
    "User experience should be our primary focus for the next quarter.",
    "I'm reading a book on system design that's really insightful.",
    "I wonder if we're overcomplicating this solution.",
    "Maybe we should step back and reconsider our overall architecture.",
    
    # Balanced viewpoints
    "While performance is important, we also need to consider maintainability.",
    "There are pros and cons to both approaches we should evaluate.",
    "I can see the merit in your suggestion, though I had a different idea.",
    "Let's consider multiple options before making a decision.",
    "I think we need to balance speed of delivery with code quality.",
    
    # Enthusiasm and interests
    "I'm really excited about the machine learning features we're adding!",
    "I spent the weekend learning Rust, it's a fascinating language.",
    "The conference last week had some great talks on microservices.",
    "I love solving complex algorithmic problems in my spare time.",
    "I've been experimenting with some new data visualization techniques.",
    
    # Values-oriented
    "Code readability is really important for long-term maintenance.",
    "We should prioritize accessibility in our UI components.",
    "Security should never be an afterthought in our development process.",
    "I believe in thorough testing before shipping to production.",
    "Documentation is crucial for onboarding new team members.",
    
    # Detail-oriented
    "I noticed that the API returns inconsistent date formats.",
    "There's a small edge case we should handle when the user has no profile.",
    "The spacing in the UI breaks at specific viewport widths.",
    "We should add validation to prevent this type of input error.",
    "I added comments to explain the complex regex pattern.",
    
    # Growth mindset
    "I'm learning a new framework to expand my skillset.",
    "Thanks for the constructive feedback, I'll work on improving that area.",
    "What would be the best way to implement this pattern?",
    "I made a mistake in my approach, let me try a different solution.",
    "I'd appreciate any resources you recommend for learning more about this.",
    
    # Analytical
    "Let me analyze the performance metrics before we make a decision.",
    "If we compare the two approaches based on these criteria...",
    "The data suggests that users prefer the simplified interface.",
    "I've identified three potential solutions to this problem.",
    "Looking at the error logs, I see a pattern emerging."
]

def create_test_data():
    """Create a test user and messages"""
    db = SessionLocal()
    
    try:
        # Check if our test user already exists
        test_user = db.query(User).filter(User.username == "test_user").first()
        
        if not test_user:
            # Create a new test user
            test_user = User(
                username="test_user",
                email="test@example.com",
                description="A test user for personality analysis"
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)
            print(f"Created test user: {test_user.username} (ID: {test_user.id})")
        else:
            print(f"Using existing test user: {test_user.username} (ID: {test_user.id})")
            
        # Create a test conversation if none exists
        conversation = db.query(Conversation).filter(
            Conversation.user_id == test_user.id
        ).first()
        
        if not conversation:
            conversation = Conversation(
                user_id=test_user.id,
                source="test"
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            print(f"Created test conversation (ID: {conversation.id})")
        else:
            print(f"Using existing conversation (ID: {conversation.id})")
            
        # Count existing messages
        existing_messages = db.query(Message).filter(
            Message.user_id == test_user.id
        ).count()
        
        print(f"Found {existing_messages} existing messages")
        
        # If we have fewer than 120 messages, add more
        messages_to_add = 120 - existing_messages
        
        if messages_to_add > 0:
            print(f"Adding {messages_to_add} new messages...")
            
            # Calculate a starting date (30 days ago)
            now = datetime.now()
            start_date = now - timedelta(days=30)
            
            # Create messages with timestamps spread over the last 30 days
            for i in range(messages_to_add):
                # Pick a random message or combine two for variety
                if random.random() < 0.3:  # 30% chance of combined message
                    content = f"{random.choice(SAMPLE_MESSAGES)} {random.choice(SAMPLE_MESSAGES)}"
                else:
                    content = random.choice(SAMPLE_MESSAGES)
                    
                # Calculate a timestamp
                message_date = start_date + timedelta(
                    seconds=random.randint(0, int((now - start_date).total_seconds()))
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
                if i % 20 == 0 or i == messages_to_add - 1:
                    db.commit()
                    print(f"Added {min(i+1, messages_to_add)} messages...")
            
            print(f"Successfully added {messages_to_add} messages")
        else:
            print("No new messages needed")
            
        # Return the test user
        return test_user
            
    finally:
        db.close()

if __name__ == "__main__":
    user = create_test_data()
    print(f"\nTest data created successfully!")
    print(f"Run this to generate a personality profile:")
    print(f"python test_personality.py test_user") 