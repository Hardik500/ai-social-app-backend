import httpx
import os
from dotenv import load_dotenv
import json
from sqlalchemy import text
from sqlalchemy.orm import Session
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
import time
import hashlib
import redis
from functools import lru_cache

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
        
        # Initialize in-memory cache with 100 items max
        self.response_cache = {}
        self.cache_size = 100
        
        # Setup Redis connection if available
        redis_url = os.getenv("REDIS_URL")
        self.redis_client = None
        self.cache_expiry = 3600  # Cache expiry in seconds (1 hour)
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                print(f"Redis cache initialized at {redis_url}")
            except Exception as e:
                print(f"Failed to connect to Redis: {str(e)}")
                self.redis_client = None
        
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
        
        # Generate embedding for description in a background task
        embedding_task = asyncio.create_task(embedding_service.generate_embedding(description))
        
        # While embedding generates, start creating profile with placeholder
        embedding = await embedding_task
        
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
        
        # Process messages in parallel batches to speed up analysis
        message_batches = self._create_message_batches(messages, 10)  # Process 10 messages per batch
        processed_messages = []
        
        # Create tasks for parallel processing
        tasks = []
        for batch in message_batches:
            tasks.append(self._process_message_batch(batch))
        
        # Execute tasks in parallel
        batch_results = await asyncio.gather(*tasks)
        
        # Combine results
        for result in batch_results:
            processed_messages.extend(result)
            
        # Join messages with separator for context
        message_text = "\n---\n".join(processed_messages)
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
    
    def _create_message_batches(self, messages: List[str], batch_size: int) -> List[List[str]]:
        """Split messages into batches for parallel processing."""
        return [messages[i:i + batch_size] for i in range(0, len(messages), batch_size)]
        
    async def _process_message_batch(self, messages: List[str]) -> List[str]:
        """Process a batch of messages - this can include any preprocessing needed."""
        # In the future, you could add more complex preprocessing here
        return messages
            
    async def generate_response(self, user_id: int, question: str, db: Session) -> Optional[str]:
        """Generate a response to a question based on a user's personality profile."""
        # Try to get from cache first
        cache_key = self._get_cache_key(user_id, question)
        cached_response = self._get_from_cache(cache_key)
        
        if cached_response:
            print(f"Cache hit for user {user_id} and question: {question[:30]}...")
            return cached_response
            
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
                        response_content = result["message"]["content"]
                        # Cache the response
                        self._add_to_cache(cache_key, response_content)
                        return response_content
                    return None
                else:
                    error_msg = f"Error generating response: {response.status_code} - {response.text}"
                    print(error_msg)
                    return None
        except Exception as e:
            print(f"Exception when calling Ollama API: {str(e)}")
            return None
            
    def _get_cache_key(self, user_id: int, question: str) -> str:
        """Generate a cache key based on user_id and question."""
        # Create a unique cache key using md5 hash of user_id and question
        key_string = f"{user_id}:{question}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
        
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Try to get a response from cache (either Redis or in-memory)."""
        # Try Redis first if available
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"personality_response:{cache_key}")
                if cached:
                    return cached.decode('utf-8')
            except Exception as e:
                print(f"Redis cache error: {str(e)}")
        
        # Fall back to in-memory cache
        return self.response_cache.get(cache_key)
        
    def _add_to_cache(self, cache_key: str, response: str) -> None:
        """Add a response to cache (both Redis and in-memory)."""
        # Add to Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"personality_response:{cache_key}", 
                    self.cache_expiry,
                    response
                )
            except Exception as e:
                print(f"Redis cache error: {str(e)}")
        
        # Add to in-memory cache
        self.response_cache[cache_key] = response
        
        # Prune cache if it gets too large
        if len(self.response_cache) > self.cache_size:
            # Remove oldest entries (simplistic approach)
            keys_to_remove = list(self.response_cache.keys())[:-self.cache_size]
            for key in keys_to_remove:
                del self.response_cache[key]
                
    # Optional: add method to preload frequent questions
    async def preload_common_questions(self, user_id: int, common_questions: List[str], db: Session) -> None:
        """Preload responses to common questions to improve response times."""
        for question in common_questions:
            asyncio.create_task(self.generate_response(user_id, question, db))
            
    async def preload_related_questions(self, user_id: int, original_question: str, original_answer: str, db: Session) -> None:
        """Generate and cache responses to questions related to the original question."""
        # Get the user's active personality profile
        profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id,
            PersonalityProfile.is_active == True
        ).first()
        
        if not profile:
            return
            
        # Create follow-up questions based on the original question and answer
        follow_up_questions = await self._generate_related_questions(
            original_question, 
            original_answer, 
            profile.system_prompt
        )
        
        # Preload responses for these questions
        for question in follow_up_questions:
            asyncio.create_task(self.generate_response(user_id, question, db))
    
    async def _generate_related_questions(self, original_question: str, original_answer: str, system_prompt: str) -> List[str]:
        """Generate follow-up questions related to the original question and answer."""
        url = f"{self.base_url}/api/chat"
        
        # Create a prompt to generate related questions
        related_questions_prompt = f"""
        Based on this conversation:
        User: {original_question}
        You: {original_answer}
        
        Generate 3 likely follow-up questions the user might ask next. 
        Return the questions as a JSON array of strings. For example:
        ["Follow-up question 1?", "Follow-up question 2?", "Follow-up question 3?"]
        """
        
        chat_messages = [
            {"role": "system", "content": "You are a helpful assistant that generates likely follow-up questions."},
            {"role": "user", "content": related_questions_prompt}
        ]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={
                        "model": self.model,
                        "messages": chat_messages,
                        "stream": False,
                        "format": "json"
                    },
                    timeout=15.0  # Shorter timeout for this task
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "message" in result and "content" in result["message"]:
                        content = result["message"]["content"]
                        try:
                            # Parse the JSON content (might be wrapped in markdown code blocks)
                            content = content.strip()
                            if content.startswith("```json"):
                                content = content[7:]
                            if content.startswith("```"):
                                content = content[3:]
                            if content.endswith("```"):
                                content = content[:-3]
                            
                            questions = json.loads(content.strip())
                            if isinstance(questions, list) and len(questions) > 0:
                                return questions[:3]  # Limit to 3 questions
                        except json.JSONDecodeError:
                            # If parsing fails, try to extract questions using regex
                            import re
                            matches = re.findall(r'"([^"]+\?)"', content)
                            if matches:
                                return matches[:3]
                    
                # Return empty list if we couldn't generate questions
                return []
        except Exception as e:
            print(f"Error generating related questions: {str(e)}")
            return []
            
    async def generate_response_stream(self, user_id: int, question: str, db: Session):
        """
        Generate a streaming response to a question based on a user's personality profile.
        This allows the client to start displaying the response as it's being generated.
        """
        # Check cache first for immediate response
        cache_key = self._get_cache_key(user_id, question)
        cached_response = self._get_from_cache(cache_key)
        
        if cached_response:
            # For cached responses, return immediately as a single chunk
            print(f"Cache hit for user {user_id} and question: {question[:30]}...")
            yield json.dumps({
                "type": "cached",
                "content": cached_response,
                "done": True
            }) + "\n"
            return
            
        # Get the user's active personality profile
        profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id,
            PersonalityProfile.is_active == True
        ).first()
        
        if not profile:
            yield json.dumps({
                "error": "No active personality profile found",
                "done": True
            }) + "\n"
            return
            
        # Get the user for their username
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            yield json.dumps({
                "error": "User not found",
                "done": True
            }) + "\n"
            return
            
        # Create chat prompt
        url = f"{self.base_url}/api/chat"
        
        chat_messages = [
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            full_response = ""
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={
                        "model": self.model,
                        "messages": chat_messages,
                        "stream": True  # Request streaming response
                    },
                    timeout=60.0,
                    headers={"Accept": "application/json"},
                )
                
                if response.status_code != 200:
                    yield json.dumps({
                        "error": f"Error generating response: {response.status_code}",
                        "done": True
                    }) + "\n"
                    return
                
                # Process the streaming response
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                if content:
                                    full_response += content
                                    yield json.dumps({
                                        "type": "token",
                                        "content": content,
                                        "done": False
                                    }) + "\n"
                        except json.JSONDecodeError:
                            pass
                
                # Send final complete message
                yield json.dumps({
                    "type": "complete",
                    "content": full_response,
                    "done": True
                }) + "\n"
                
                # Cache the complete response
                if full_response:
                    self._add_to_cache(cache_key, full_response)
                
        except Exception as e:
            print(f"Exception when streaming response: {str(e)}")
            yield json.dumps({
                "error": str(e),
                "done": True
            }) + "\n"

personality_service = PersonalityService() 