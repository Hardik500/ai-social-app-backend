import httpx
import os
from dotenv import load_dotenv
import json
from sqlalchemy import text
from sqlalchemy.orm import Session
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional

from app.models.user import User
from app.models.conversation import Message
from app.models.personality import PersonalityProfile
from app.services.embedding_service import embedding_service
from app.core.prompt_manager import prompt_manager

load_dotenv()

class PersonalityService:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
        
    async def generate_profile(self, user_id: int, db: Session) -> Optional[PersonalityProfile]:
        """Generate a personality profile for a user based on their messages."""
        # Get the user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            print(f"User with ID {user_id} not found")
            return None
            
        # Get all messages by this user
        messages = db.query(Message).filter(Message.user_id == user_id).all()
        message_count = len(messages)
        print(f"Found {message_count} messages for user {user.username} (ID: {user_id}, Email: {user.email})")
        
        if not messages or message_count < 5:  # Require at least 5 messages for a good profile
            print(f"Insufficient messages for user {user.username}: {message_count} (Need at least 5)")
            return None
            
        # If there are too many messages, sample a subset to avoid timeout issues
        MAX_MESSAGES = 50  # Limit to prevent timeouts or context size issues
        
        if message_count > MAX_MESSAGES:
            import random
            print(f"Sampling {MAX_MESSAGES} messages from {message_count} total messages")
            messages = random.sample(messages, MAX_MESSAGES)
        
        # Concatenate messages into a single document
        message_texts = [msg.content for msg in messages]
        
        # Get the analysis system prompt from prompt manager
        system_prompt = prompt_manager.get_template("personality_analysis")
        
        # Request personality analysis from LLM
        traits_and_description = await self._generate_analysis(message_texts, system_prompt)
        
        if traits_and_description is None:
            print(f"Failed to generate personality analysis for user {user.username} (ID: {user_id})")
            return None
        
        # Extract traits and summary from the response
        traits = traits_and_description.get("traits", {})
        communication_style = traits_and_description.get("communication_style", {})
        interests = traits_and_description.get("interests", [])
        values = traits_and_description.get("values", [])
        summary = traits_and_description.get("summary", "")
        
        # Format the description using the template
        description = prompt_manager.format_template(
            "description_template",
            summary=summary,
            openness=traits.get('openness', 'N/A'),
            conscientiousness=traits.get('conscientiousness', 'N/A'),
            extraversion=traits.get('extraversion', 'N/A'),
            agreeableness=traits.get('agreeableness', 'N/A'),
            neuroticism=traits.get('neuroticism', 'N/A'),
            communication_style=communication_style if isinstance(communication_style, str) else 
                                ', '.join([f'{k}: {v}' for k, v in communication_style.items()]) 
                                if isinstance(communication_style, dict) else communication_style,
            interests=interests if isinstance(interests, str) else 
                      ', '.join(interests) if isinstance(interests, list) else interests,
            values=values if isinstance(values, str) else 
                   ', '.join(values) if isinstance(values, list) else values
        )
        
        # Format the system prompt for simulation using the template
        system_prompt = prompt_manager.format_template(
            "personality_simulation",
            username=user.username,
            openness=traits.get('openness', 5),
            conscientiousness=traits.get('conscientiousness', 5),
            extraversion=traits.get('extraversion', 5),
            agreeableness=traits.get('agreeableness', 5),
            neuroticism=traits.get('neuroticism', 5),
            communication_style=communication_style if isinstance(communication_style, str) else 
                                ', '.join([f'{k}: {v}' for k, v in communication_style.items()]) 
                                if isinstance(communication_style, dict) else communication_style,
            interests=interests if isinstance(interests, str) else 
                      ', '.join(interests) if isinstance(interests, list) else interests,
            values=values if isinstance(values, str) else 
                   ', '.join(values) if isinstance(values, list) else values,
            summary=summary
        )
        
        # Generate embedding for description (for similarity search later)
        embedding = await embedding_service.generate_embedding(description)
        
        # Create new personality profile
        new_profile = PersonalityProfile(
            user_id=user_id,
            traits=traits_and_description,
            description=description,
            embedding=embedding,
            message_count=message_count,
            system_prompt=system_prompt,
            is_active=True
        )
        
        # Set any existing profiles to inactive
        existing_profiles = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id,
            PersonalityProfile.is_active == True
        ).all()
        
        for profile in existing_profiles:
            profile.is_active = False
            
        # Save the new profile
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        
        return new_profile
        
    async def _generate_analysis(self, messages: List[str], system_prompt: str) -> Optional[Dict[str, Any]]:
        """Generate a personality analysis using the LLM."""
        url = f"{self.base_url}/api/chat"
        
        # Join messages with separator for context
        message_text = "\n---\n".join(messages)
        print(f"Analyzing {len(messages)} messages with total length of {len(message_text)} characters")
        
        # Create chat prompt
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are some message samples from a user:\n\n{message_text}\n\nCreate a detailed personality profile."}
        ]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={
                        "model": self.model,
                        "messages": chat_messages,
                        "stream": False,  # Request non-streaming response
                        "format": "json"  # Request JSON output format
                    },
                    timeout=120.0  # Longer timeout for analysis (2 minutes)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract the content from the response
                    if "message" in result and "content" in result["message"]:
                        content = result["message"]["content"]
                        try:
                            # Parse the JSON content (might be wrapped in markdown code blocks)
                            content = content.strip()
                            if content.startswith("```json"):
                                content = content[7:]
                            elif content.startswith("```"):
                                content = content[3:]
                            if content.endswith("```"):
                                content = content[:-3]
                            
                            parsed_content = json.loads(content.strip())
                            return parsed_content
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON from response: {content}")
                            print(f"JSON Error: {str(e)}")
                            return None
                    else:
                        print(f"Unexpected API response format: {result}")
                        return None
                else:
                    error_msg = f"Error generating analysis: {response.status_code} - {response.text}"
                    print(error_msg)
                    return None
        except Exception as e:
            print(f"Exception when calling Ollama API: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
            
    async def generate_response(self, user_id: int, question: str, db: Session) -> Optional[str]:
        """Generate a response to a question based on a user's personality profile."""
        # Get the user's active personality profile
        profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id,
            PersonalityProfile.is_active == True
        ).first()
        
        if not profile:
            return None
            
        # Get the user for their username
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None
            
        # Create chat prompt
        url = f"{self.base_url}/api/chat"
        
        chat_messages = [
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={
                        "model": self.model,
                        "messages": chat_messages,
                        "stream": False  # Request non-streaming response
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract the content from the response
                    if "message" in result and "content" in result["message"]:
                        return result["message"]["content"]
                    return None
                else:
                    error_msg = f"Error generating response: {response.status_code} - {response.text}"
                    print(error_msg)
                    return None
        except Exception as e:
            print(f"Exception when calling Ollama API: {str(e)}")
            return None

# Create singleton instance
personality_service = PersonalityService() 