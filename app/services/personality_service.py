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
import re

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
        
        # Check if there's an existing profile we can update incrementally
        existing_profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id, 
            PersonalityProfile.is_active == True
        ).first()
        
        if existing_profile and hasattr(existing_profile, 'last_message_id') and existing_profile.last_message_id:
            # Attempt incremental update if there are new messages
            try:
                # Get only new messages since last profile update
                new_messages = db.query(Message).filter(
                    Message.user_id == user_id,
                    Message.id > existing_profile.last_message_id
                ).all()
                
                # If there are enough new messages, do an incremental update
                if len(new_messages) >= 5:
                    print(f"Performing incremental update with {len(new_messages)} new messages")
                    return await self.update_profile_incrementally(
                        user_id, existing_profile, new_messages, db
                    )
            except Exception as e:
                print(f"Error during incremental update check: {str(e)}")
                # Fall back to full profile generation
                pass
            
        # If there are too many messages, sample a subset to avoid timeout issues
        MAX_MESSAGES = 1000  # Increased from 50 to handle more messages for initial profile
        
        if message_count > MAX_MESSAGES:
            # Use embedding-guided selection instead of random sampling
            selected_messages = self._select_representative_messages(messages, MAX_MESSAGES, db)
            print(f"Selected {len(selected_messages)} representative messages from {message_count} total messages")
            messages = selected_messages
        
        # Extract example messages for the simulation prompt
        example_messages = self._extract_example_messages(messages)
        
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
            summary=summary,
            # Use extracted examples if available, otherwise use defaults
            example_question_1=example_messages.get("question1", "How are you today?"),
            example_response_1=example_messages.get("response1", "I'm doing well, thanks for asking!"),
            example_question_2=example_messages.get("question2", "What do you think about this project?"),
            example_response_2=example_messages.get("response2", "I think it's interesting and has a lot of potential."),
            topic="general conversation",
            participants=f"{user.username} and others",
            mood="neutral"
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
            is_active=True,
            last_message_id=max(msg.id for msg in messages) if messages else None
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
        
        # Process messages in smaller batches to reduce context size
        max_messages_per_request = 50
        
        if len(messages) > max_messages_per_request:
            print(f"Processing {len(messages)} messages in smaller batches of {max_messages_per_request}")
            # Split into chunks of max_messages_per_request
            message_chunks = [messages[i:i + max_messages_per_request] 
                             for i in range(0, len(messages), max_messages_per_request)]
            
            # Process each chunk and collect results
            chunk_results = []
            for i, chunk in enumerate(message_chunks):
                print(f"Processing chunk {i+1}/{len(message_chunks)} with {len(chunk)} messages")
                result = await self._generate_chunk_analysis(chunk, system_prompt, is_chunk=True)
                if result:
                    chunk_results.append(result)
                else:
                    print(f"Failed to process chunk {i+1}")
                    
            if not chunk_results:
                return None
                
            # Merge the chunk results
            merged_result = self._merge_chunk_results(chunk_results)
            return merged_result
        
        # For small message batches, process directly
        return await self._generate_chunk_analysis(messages, system_prompt, is_chunk=False)
    
    async def _generate_chunk_analysis(self, messages: List[str], system_prompt: str, is_chunk: bool) -> Optional[Dict[str, Any]]:
        """Generate analysis for a chunk of messages."""
        url = f"{self.base_url}/api/chat"
        
        # Process messages in parallel batches to speed up analysis
        message_batches = self._create_message_batches(messages, 5)  # Smaller batch size for faster processing
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
        user_prompt = f"Here are some message samples from a user:\n\n{message_text}\n\n"
        if is_chunk:
            user_prompt += "Create a partial personality profile based on these messages only."
        else:
            user_prompt += "Create a detailed personality profile."
            
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
                    timeout=180.0  # Increased timeout for analysis (3 minutes)
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
                            
                            # Clean up the content
                            content = content.strip()
                            
                            try:
                                # Try simple JSON parsing first
                                parsed_content = json.loads(content)
                                return parsed_content
                            except json.JSONDecodeError as e:
                                print(f"Initial JSON parsing failed: {str(e)}")
                                print("Attempting to fix the JSON structure...")
                                
                                # Create a fallback structure
                                fallback = {
                                    "traits": {
                                        "openness": 5,
                                        "conscientiousness": 5,
                                        "extraversion": 5,
                                        "agreeableness": 5,
                                        "neuroticism": 5
                                    },
                                    "communication_style": {"style": "neutral"},
                                    "interests": [],
                                    "values": [],
                                    "summary": "Unable to generate complete profile due to processing error."
                                }
                                
                                # Try to extract traits using string operations instead of regex
                                try:
                                    if '"traits"' in content:
                                        traits_start = content.find('"traits"') + 8
                                        traits_start = content.find('{', traits_start)
                                        if traits_start > 0:
                                            traits_end = content.find('}', traits_start) + 1
                                            if traits_end > traits_start:
                                                traits_json = content[traits_start:traits_end]
                                                try:
                                                    traits = json.loads(traits_json)
                                                    for key, value in traits.items():
                                                        if isinstance(value, (int, float)) and 1 <= value <= 10:
                                                            fallback["traits"][key.lower()] = value
                                                except:
                                                    pass
                                except Exception as ex:
                                    print(f"Error extracting traits: {str(ex)}")
                                
                                # Try to extract interests
                                try:
                                    if '"interests"' in content:
                                        interests_start = content.find('"interests"') + 11
                                        interests_start = content.find('[', interests_start)
                                        if interests_start > 0:
                                            interests_end = content.find(']', interests_start) + 1
                                            if interests_end > interests_start:
                                                interests_json = content[interests_start:interests_end]
                                                try:
                                                    interests = json.loads(interests_json)
                                                    if isinstance(interests, list):
                                                        fallback["interests"] = [i for i in interests if isinstance(i, str)]
                                                except:
                                                    pass
                                except Exception as ex:
                                    print(f"Error extracting interests: {str(ex)}")
                                
                                # Try to extract values
                                try:
                                    if '"values"' in content:
                                        values_start = content.find('"values"') + 8
                                        values_start = content.find('[', values_start)
                                        if values_start > 0:
                                            values_end = content.find(']', values_start) + 1
                                            if values_end > values_start:
                                                values_json = content[values_start:values_end]
                                                try:
                                                    values = json.loads(values_json)
                                                    if isinstance(values, list):
                                                        fallback["values"] = [v for v in values if isinstance(v, str)]
                                                except:
                                                    pass
                                except Exception as ex:
                                    print(f"Error extracting values: {str(ex)}")
                                    
                                print("Using fallback structure with extracted data")
                                return fallback
                                
                        except Exception as e:
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
    
    def _select_representative_messages(self, messages: List[Message], max_count: int, db: Session) -> List[Message]:
        """Select representative messages using embedding-based clustering and temporal weighting."""
        # Always use time-weighted selection for now since k-means may not be available
        return self._select_time_weighted_messages(messages, max_count)
    
    def _select_time_weighted_messages(self, messages: List[Message], max_count: int) -> List[Message]:
        """Select messages with exponential time weighting to prioritize recent messages."""
        import random
        import math
        from datetime import datetime
        
        now = datetime.utcnow().timestamp()
        
        # Apply exponential decay weighting based on message age
        weighted_messages = []
        for msg in messages:
            msg_time = msg.created_at.timestamp()
            time_diff_days = (now - msg_time) / (24 * 3600)  # Convert to days
            weight = math.exp(-0.001 * time_diff_days)  # Exponential decay
            weighted_messages.append((msg, weight))
        
        # Ensure we always include some of the most recent messages
        recent_count = min(max_count // 4, len(messages))
        recent_messages = sorted(messages, key=lambda m: m.created_at, reverse=True)[:recent_count]
        recent_ids = {msg.id for msg in recent_messages}
        
        # Do weighted random sampling for the remaining slots
        remaining_slots = max_count - recent_count
        remaining_messages = [m for m in messages if m.id not in recent_ids]
        
        if not remaining_messages:
            return recent_messages
            
        # Weighted random sampling
        weights = [math.exp(-0.001 * (now - msg.created_at.timestamp()) / (24 * 3600)) 
                   for msg in remaining_messages]
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Perform weighted sampling
        sampled_indices = random.choices(
            range(len(remaining_messages)), 
            weights=normalized_weights, 
            k=min(remaining_slots, len(remaining_messages))
        )
        
        # Combine recent and sampled messages
        selected_messages = recent_messages + [remaining_messages[i] for i in sampled_indices]
        
        return selected_messages
        
    async def _process_message_batch(self, messages: List[str]) -> List[str]:
        """Process a batch of messages - this can include any preprocessing needed."""
        # In the future, you could add more complex preprocessing here
        return messages
            
    async def generate_response(self, user_id: int, question: str, db: Session) -> Optional[str]:
        """Generate a response to a question based on a user's personality profile, conversation history, and user information."""
        print(f"Generating response for user {user_id} and question: {question[:30]}...")
        
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
            print(f"No active profile found for user {user_id}")
            return None
            
        # Get the user for their username and other information
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            print(f"No user found with ID {user_id}")
            return None

        print(f"Found user {user.username} with active profile")

        # Get recent conversations (last 50 messages)
        recent_messages = (
            db.query(Message)
            .filter(Message.user_id == user_id)
            .order_by(Message.created_at.desc())
            .limit(50)
            .all()
        )
        print(f"Retrieved {len(recent_messages)} recent messages for context")

        # Format user information and conversation history
        user_context = f"\nUser Information:\n"
        user_context += f"- Username: {user.username}\n"
        if hasattr(user, 'email') and user.email:
            user_context += f"- Email: {user.email}\n"
        if hasattr(user, 'full_name') and user.full_name:
            user_context += f"- Full Name: {user.full_name}\n"
        if hasattr(user, 'created_at') and user.created_at:
            user_context += f"- Member since: {user.created_at.strftime('%Y-%m-%d')}\n"
        if hasattr(profile, 'message_count') and profile.message_count:
            user_context += f"- Total messages: {profile.message_count}\n"

        # Format conversation history
        conversation_history = ""
        if recent_messages:
            conversation_history = "\nRecent conversation history:\n"
            for msg in reversed(recent_messages):  # Show in chronological order
                # Add timestamp if available
                timestamp = f" ({msg.created_at.strftime('%Y-%m-%d %H:%M')})" if hasattr(msg, 'created_at') and msg.created_at else ""
                conversation_history += f"- {msg.content}{timestamp}\n"
            
        # Create chat prompt
        url = f"{self.base_url}/api/chat"
        
        # Enhance the system prompt with user info, question-specific context and conversation history
        system_prompt = self._enhance_system_prompt_for_question(
            profile.system_prompt + user_context + conversation_history, 
            user.username, 
            question
        )
        
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        print(f"Calling Ollama API for user {user.username} with model {self.model}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={
                        "model": self.model,
                        "messages": chat_messages,
                        "stream": False  # Request non-streaming response
                    },
                    timeout=120.0  # Increase timeout to 2 minutes
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract the content from the response
                    if "message" in result and "content" in result["message"]:
                        response_content = result["message"]["content"]
                        print(f"Successfully generated response for user {user.username}")
                        # Cache the response
                        self._add_to_cache(cache_key, response_content)
                        return response_content
                    print(f"Invalid response format from Ollama API for user {user.username}")
                    return None
                else:
                    error_msg = f"Error generating response: {response.status_code} - {response.text}"
                    print(error_msg)
                    return None
        except Exception as e:
            print(f"Exception when calling Ollama API for user {user.username}: {str(e)}")
            return None
            
    def _enhance_system_prompt_for_question(self, system_prompt: str, username: str, question: str) -> str:
        """
        Enhance the system prompt with question-specific context.
        This adapts the existing system prompt to the current conversation.
        """
        # Detect question category/topic
        topic = "general conversation"
        mood = "neutral"
        
        # Simple heuristics to determine topic and mood
        question_lower = question.lower()
        
        # Detect topics
        if any(word in question_lower for word in ["work", "job", "career", "project", "task"]):
            topic = "work-related discussion"
        elif any(word in question_lower for word in ["personal", "family", "friend", "life", "relationship"]):
            topic = "personal conversation"
        elif any(word in question_lower for word in ["advice", "suggest", "recommend", "help"]):
            topic = "advice seeking"
        elif any(word in question_lower for word in ["opinion", "think", "feel", "believe"]):
            topic = "opinion sharing"
        elif any(word in question_lower for word in ["explain", "what is", "how does", "why"]):
            topic = "explanation request"
            
        # Detect mood
        if any(word in question_lower for word in ["urgent", "emergency", "critical", "asap", "quickly"]):
            mood = "urgent"
        elif any(word in question_lower for word in ["sad", "sorry", "upset", "disappointed"]):
            mood = "empathetic"
        elif any(word in question_lower for word in ["excited", "happy", "great", "awesome"]):
            mood = "enthusiastic"
        elif any(word in question_lower for word in ["confused", "don't understand", "unclear"]):
            mood = "clarifying"
            
        # Update the context in the system prompt
        updated_prompt = system_prompt
        
        try:
            # Try to update topic, participants and mood if they exist in the prompt
            if "Current conversation context:" in system_prompt:
                # Update topic if present
                topic_pattern = r"- Topic: .*"
                if re.search(topic_pattern, system_prompt):
                    updated_prompt = re.sub(topic_pattern, f"- Topic: {topic}", updated_prompt)
                
                # Update participants if present
                participants_pattern = r"- Participants: .*"
                if re.search(participants_pattern, system_prompt):
                    updated_prompt = re.sub(participants_pattern, f"- Participants: {username} and the person asking", updated_prompt)
                
                # Update mood if present
                mood_pattern = r"- Mood: .*"
                if re.search(mood_pattern, system_prompt):
                    updated_prompt = re.sub(mood_pattern, f"- Mood: {mood}", updated_prompt)
            
            # Add a reminder about the current question
            if not updated_prompt.endswith("\n"):
                updated_prompt += "\n"
            updated_prompt += f"\nCurrent question context: You are now responding to a question about '{topic}' with a '{mood}' tone."
        except Exception as e:
            # If anything fails, just use the original prompt
            print(f"Error enhancing system prompt: {str(e)}")
            return system_prompt
            
        return updated_prompt
        
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
        """Generate related follow-up questions."""
        url = f"{self.base_url}/api/chat"
        
        # Create a specialized system prompt for generating follow-up questions
        followup_system_prompt = "You are a helpful assistant tasked with generating natural follow-up questions. " + \
                               "Based on the conversation, suggest questions that would naturally continue the discussion " + \
                               "in a way that's consistent with the personality profile. Focus on questions that are " + \
                               "relevant to the previous exchange and would elicit informative or interesting responses."
        
        # Create prompt
        user_prompt = f"""Based on the following question and answer, suggest 3 related follow-up questions that the user might want to ask next.
        
Question: {original_question}
Answer: {original_answer}

The answer was generated based on this personality profile:
{system_prompt[:500]}...

Return only the questions as a JSON array of strings. Make sure the questions are:
1. Relevant to the conversation context
2. Consistent with the natural flow of the conversation
3. Likely to elicit an informative response
4. Varied in their focus to explore different aspects of the topic"""
        
        chat_messages = [
            {"role": "system", "content": followup_system_prompt},
            {"role": "user", "content": user_prompt}
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
                    timeout=10.0
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
        Generate a streaming response to a question based on a user's personality profile and recent conversation history.
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

        # Get recent conversations (last 50 messages)
        recent_messages = (
            db.query(Message)
            .filter(Message.user_id == user_id)
            .order_by(Message.created_at.desc())
            .limit(50)
            .all()
        )

        # Format conversation history
        conversation_history = ""
        if recent_messages:
            conversation_history = "\nRecent conversation history:\n"
            for msg in reversed(recent_messages):  # Show in chronological order
                conversation_history += f"- {msg.content}\n"
            
        # Create chat prompt
        url = f"{self.base_url}/api/chat"
        
        # Enhance the system prompt with question-specific context and conversation history
        system_prompt = self._enhance_system_prompt_for_question(
            profile.system_prompt + conversation_history, 
            user.username, 
            question
        )
        
        chat_messages = [
            {"role": "system", "content": system_prompt},
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

    async def update_profile_incrementally(
        self, user_id: int, existing_profile: PersonalityProfile, 
        new_messages: List[Message], db: Session
    ) -> PersonalityProfile:
        """Update an existing personality profile incrementally with new messages."""
        MAX_DELTA_MESSAGES = 50  # Maximum number of new messages to process at once
        
        # If too many new messages, select representative ones
        if len(new_messages) > MAX_DELTA_MESSAGES:
            selected_messages = self._select_representative_messages(new_messages, MAX_DELTA_MESSAGES, db)
            print(f"Selected {len(selected_messages)} representative new messages from {len(new_messages)} total new")
            new_messages = selected_messages
            
        # Get the message texts
        message_texts = [msg.content for msg in new_messages]
        
        # Extract some example messages for the simulation prompt
        example_messages = self._extract_example_messages(new_messages)
        
        # Get existing traits from profile
        existing_traits = existing_profile.traits
        
        # Get the incremental update system prompt
        system_prompt = prompt_manager.get_template("personality_incremental_update")
        
        # Format the system prompt with existing traits
        formatted_prompt = system_prompt.format(
            existing_profile=json.dumps(existing_traits, indent=2),
            message_count=len(new_messages)
        )
        
        # Generate delta analysis from LLM
        traits_delta = await self._generate_analysis(message_texts, formatted_prompt)
        
        if traits_delta is None:
            print(f"Failed to generate incremental analysis, falling back to full profile generation")
            # Make existing profile inactive
            existing_profile.is_active = False
            db.commit()
            # Generate full profile
            return await self.generate_profile(user_id, db)
            
        # Merge the existing traits with delta updates
        updated_traits = self._merge_traits(existing_traits, traits_delta)
        
        # Get the latest message ID for future incremental updates
        latest_message_id = max(msg.id for msg in new_messages) if new_messages else existing_profile.last_message_id
        
        # Update the total message count
        total_message_count = existing_profile.message_count + len(new_messages)
        
        # Extract updated components
        summary = updated_traits.get("summary", "")
        traits = updated_traits.get("traits", {})
        communication_style = updated_traits.get("communication_style", {})
        interests = updated_traits.get("interests", [])
        values = updated_traits.get("values", [])
        
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
        
        # Get the username for this user_id
        username = db.query(User).filter(User.id == user_id).first().username
        
        # Format the system prompt for simulation
        system_prompt = prompt_manager.format_template(
            "personality_simulation",
            username=username,
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
            summary=summary,
            # Use extracted examples if available, otherwise use defaults
            example_question_1=example_messages.get("question1", "How are you today?"),
            example_response_1=example_messages.get("response1", "I'm doing well, thanks for asking!"),
            example_question_2=example_messages.get("question2", "What do you think about this project?"),
            example_response_2=example_messages.get("response2", "I think it's interesting and has a lot of potential."),
            topic="general conversation",
            participants=f"{username} and others",
            mood="neutral"
        )
        
        # Generate embedding for description
        embedding = await embedding_service.generate_embedding(description)
        
        # Create a change log entry
        change_log = {
            "timestamp": time.time(),
            "new_message_count": len(new_messages),
            "total_message_count": total_message_count,
            "changes": traits_delta.get("changes", {})
        }
        
        # Add the change log to the existing profile if it exists, otherwise create it
        if hasattr(existing_profile, 'change_log') and existing_profile.change_log:
            if isinstance(existing_profile.change_log, str):
                try:
                    existing_change_log = json.loads(existing_profile.change_log)
                except:
                    existing_change_log = []
            else:
                existing_change_log = existing_profile.change_log
            
            if not isinstance(existing_change_log, list):
                existing_change_log = [existing_change_log]
                
            existing_change_log.append(change_log)
            change_log_json = json.dumps(existing_change_log)
        else:
            change_log_json = json.dumps([change_log])
            
        # Get delta embedding if available
        delta_embedding = None
        try:
            delta_text = traits_delta.get("delta_summary", "")
            if delta_text:
                delta_embedding = await embedding_service.generate_embedding(delta_text)
        except:
            delta_embedding = None
            
        # Update existing profile
        existing_profile.traits = updated_traits
        existing_profile.description = description
        existing_profile.embedding = embedding
        existing_profile.message_count = total_message_count
        existing_profile.system_prompt = system_prompt
        existing_profile.last_message_id = latest_message_id
        
        # Add delta embedding and change log if supported
        if hasattr(existing_profile, 'delta_embeddings') and delta_embedding:
            # Append to existing delta embeddings if they exist
            if existing_profile.delta_embeddings:
                if isinstance(existing_profile.delta_embeddings, str):
                    try:
                        existing_delta_embeddings = json.loads(existing_profile.delta_embeddings)
                    except:
                        existing_delta_embeddings = []
                else:
                    existing_delta_embeddings = existing_profile.delta_embeddings
                
                if not isinstance(existing_delta_embeddings, list):
                    existing_delta_embeddings = [existing_delta_embeddings] 
                
                existing_delta_embeddings.append(delta_embedding)
                existing_profile.delta_embeddings = existing_delta_embeddings
            else:
                existing_profile.delta_embeddings = [delta_embedding]
        
        if hasattr(existing_profile, 'change_log'):
            existing_profile.change_log = change_log_json
            
        # Save changes
        db.commit()
        db.refresh(existing_profile)
        
        return existing_profile
        
    def _merge_traits(self, existing_traits: Dict[str, Any], delta_traits: Dict[str, Any]) -> Dict[str, Any]:
        """Merge existing traits with delta updates."""
        # Deep copy to avoid modifying the original
        updated_traits = json.loads(json.dumps(existing_traits))
        
        # Update main personality traits (OCEAN)
        if "traits" in delta_traits and "traits" in updated_traits:
            for trait, value in delta_traits["traits"].items():
                updated_traits["traits"][trait] = value
                
        # Update communication style
        if "communication_style" in delta_traits:
            if isinstance(delta_traits["communication_style"], dict) and isinstance(updated_traits.get("communication_style", {}), dict):
                for key, value in delta_traits["communication_style"].items():
                    updated_traits.setdefault("communication_style", {})[key] = value
            else:
                updated_traits["communication_style"] = delta_traits["communication_style"]
                
        # Update interests (might be a list or a string)
        if "interests" in delta_traits:
            if isinstance(delta_traits["interests"], list) and isinstance(updated_traits.get("interests", []), list):
                # Combine lists and remove duplicates while preserving order
                seen = set()
                combined = []
                for item in delta_traits["interests"] + updated_traits.get("interests", []):
                    if item.lower() not in seen:
                        combined.append(item)
                        seen.add(item.lower())
                updated_traits["interests"] = combined[:15]  # Limit to top 15
            else:
                updated_traits["interests"] = delta_traits["interests"]
                
        # Update values (might be a list or a string)
        if "values" in delta_traits:
            if isinstance(delta_traits["values"], list) and isinstance(updated_traits.get("values", []), list):
                # Combine lists and remove duplicates while preserving order
                seen = set()
                combined = []
                for item in delta_traits["values"] + updated_traits.get("values", []):
                    if item.lower() not in seen:
                        combined.append(item)
                        seen.add(item.lower())
                updated_traits["values"] = combined[:10]  # Limit to top 10
            else:
                updated_traits["values"] = delta_traits["values"]
                
        # Update summary
        if "summary" in delta_traits:
            updated_traits["summary"] = delta_traits["summary"]
            
        return updated_traits

    def _merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple chunk results into a single coherent personality profile."""
        if not chunk_results:
            return {}
            
        # Use the first chunk as a base
        merged = json.loads(json.dumps(chunk_results[0]))
        
        # Count trait values to compute average
        trait_counts = {}
        trait_mappings = {
            'openness': ['openness', 'openness_to_experience'],
            'conscientiousness': ['conscientiousness'],
            'extraversion': ['extraversion', 'extroversion'],
            'agreeableness': ['agreeableness'],
            'neuroticism': ['neuroticism']
        }
        
        # Initialize trait counts
        for canonical_name in trait_mappings.keys():
            trait_counts[canonical_name] = []
            
        # Collect all interests and values
        all_interests = set()
        all_values = set()
        communication_aspects = {}
        
        # Process all chunks
        for chunk in chunk_results:
            # Collect traits
            if "traits" in chunk:
                for trait_name, value in chunk["traits"].items():
                    if isinstance(value, (int, float)) and 1 <= value <= 10:
                        # Map variant trait names to canonical names
                        canonical_name = None
                        for canon, variants in trait_mappings.items():
                            if trait_name.lower() in variants or trait_name.lower() == canon:
                                canonical_name = canon
                                break
                        
                        if canonical_name:
                            trait_counts[canonical_name].append(value)
            
            # Collect interests - FIX: Check for string type before adding to set
            if "interests" in chunk and isinstance(chunk["interests"], list):
                for interest in chunk["interests"]:
                    if isinstance(interest, str):  # Only add if it's a string
                        all_interests.add(interest)
                    elif isinstance(interest, dict) and "name" in interest:
                        # Handle case where interest is a dict with a name field
                        all_interests.add(interest["name"])
            
            # Collect values - FIX: Check for string type before adding to set
            if "values" in chunk and isinstance(chunk["values"], list):
                for value in chunk["values"]:
                    if isinstance(value, str):  # Only add if it's a string
                        all_values.add(value)
                    elif isinstance(value, dict) and "name" in value:
                        # Handle case where value is a dict with a name field
                        all_values.add(value["name"])
                    
            # Collect communication style aspects
            if "communication_style" in chunk and isinstance(chunk["communication_style"], dict):
                for aspect, value in chunk["communication_style"].items():
                    if aspect not in communication_aspects:
                        communication_aspects[aspect] = []
                    communication_aspects[aspect].append(value)
        
        # Average out the traits
        if "traits" not in merged:
            merged["traits"] = {}
        
        for trait_name, values in trait_counts.items():
            if values:
                merged["traits"][trait_name] = round(sum(values) / len(values))
                
        # Combine interests and values
        merged["interests"] = list(all_interests)[:15]  # Limit to top 15
        merged["values"] = list(all_values)[:10]  # Limit to top 10
        
        # Merge communication style
        if "communication_style" not in merged:
            merged["communication_style"] = {}
            
        for aspect, values in communication_aspects.items():
            if isinstance(values[0], (int, float)):
                # For numeric values, take average
                merged["communication_style"][aspect] = sum(values) / len(values)
            else:
                # For text values, take most common
                from collections import Counter
                counter = Counter(values)
                merged["communication_style"][aspect] = counter.most_common(1)[0][0]
        
        # Create a combined summary
        if "summary" not in merged:
            merged["summary"] = ""
            
        summaries = [chunk.get("summary", "") for chunk in chunk_results if "summary" in chunk]
        if summaries:
            merged["summary"] = "This individual " + " ".join(summaries)
            
        return merged

    def _extract_example_messages(self, messages: List[Message]) -> Dict[str, str]:
        """Extract example messages from recent messages for use in the personality simulation prompt."""
        example_messages = {}
        
        # Default examples in case we don't have enough messages
        example_messages = {
            "question1": "How are you today?",
            "response1": "I'm doing well, thanks for asking!",
            "question2": "What do you think about this project?",
            "response2": "I think it's interesting and has a lot of potential."
        }
        
        # Sort messages by creation time (newest first)
        recent_messages = sorted(messages, key=lambda m: m.created_at, reverse=True)
        
        # Need at least 2 messages to create examples
        if len(recent_messages) >= 2:
            # Use the most recent message as the first example
            example_messages["response1"] = recent_messages[0].content
            # Generate a plausible question that could have prompted this response
            example_messages["question1"] = self._generate_plausible_question(recent_messages[0].content)
            
            # Use the second most recent message as the second example
            example_messages["response2"] = recent_messages[1].content
            # Generate a plausible question for the second example
            example_messages["question2"] = self._generate_plausible_question(recent_messages[1].content)
            
        return example_messages
        
    def _generate_plausible_question(self, message_content: str) -> str:
        """Generate a plausible question that could have prompted the given message."""
        # Simple heuristics to generate a question
        
        # If the message starts with a greeting, the question was probably "How are you?"
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if any(message_content.lower().startswith(greeting) for greeting in greetings):
            return "How are you doing today?"
            
        # If the message contains "thanks" or "thank you", the question was probably asking for help
        if "thanks" in message_content.lower() or "thank you" in message_content.lower():
            return "Could you help me with this issue?"
            
        # If the message is explaining something, the question was probably asking about it
        if "because" in message_content.lower() or "since" in message_content.lower():
            return "Can you explain why this is happening?"
            
        # If the message contains an opinion, the question was probably asking for one
        opinion_words = ["think", "believe", "opinion", "feel", "view"]
        if any(word in message_content.lower() for word in opinion_words):
            return "What do you think about this?"
            
        # Default question for other cases
        return "What's your perspective on this situation?"

    async def build_rag_enhanced_system_prompt(
        self, user_id: int, username: str, relevant_messages: List[Dict], question: str, db: Session
    ) -> str:
        """
        Build a system prompt enhanced with retrieved context messages using the personality simulation template.
        
        Args:
            user_id: The user's ID
            username: The user's username
            relevant_messages: List of relevant messages with similarity scores
            question: The current question being asked
            db: Database session
            
        Returns:
            A formatted system prompt with personality details and relevant context
        """
        # Get the user's active personality profile
        profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id,
            PersonalityProfile.is_active == True
        ).first()
        
        if not profile or not profile.traits:
            # Fallback to a generic prompt if no profile exists
            return f"You are roleplaying as {username}. Answer the following question as if you were this person."
        
        # Extract profile components
        traits = profile.traits.get("traits", {})
        communication_style = profile.traits.get("communication_style", {})
        interests = profile.traits.get("interests", [])
        values = profile.traits.get("values", [])
        summary = profile.traits.get("summary", "")
        
        # Extract example messages from the relevant messages
        example_messages = {}
        
        # Default examples in case we don't have enough messages
        example_messages = {
            "question1": "How are you today?",
            "response1": "I'm doing well, thanks for asking!",
            "question2": "What do you think about this project?",
            "response2": "I think it's interesting and has a lot of potential."
        }
        
        # Try to use relevant messages as examples if available
        if relevant_messages and len(relevant_messages) >= 2:
            # Use the most relevant message as the first example
            example_messages["response1"] = relevant_messages[0]["message"]
            # Generate a plausible question that could have prompted this response
            example_messages["question1"] = self._generate_plausible_question(relevant_messages[0]["message"])
            
            # Use the second most relevant message as the second example
            example_messages["response2"] = relevant_messages[1]["message"]
            # Generate a plausible question for the second example
            example_messages["question2"] = self._generate_plausible_question(relevant_messages[1]["message"])
        
        # Format the system prompt for simulation using the template
        system_prompt = prompt_manager.format_template(
            "personality_simulation",
            username=username,
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
            summary=summary,
            example_question_1=example_messages["question1"],
            example_response_1=example_messages["response1"],
            example_question_2=example_messages["question2"],
            example_response_2=example_messages["response2"],
            topic=f"answering '{question}'",
            participants=f"{username} and the person asking the question",
            mood="helpful"
        )
        
        # Append relevant message context if available
        if relevant_messages:
            context_texts = []
            for i, msg in enumerate(relevant_messages):
                context_texts.append(f"Message {i+1}: \"{msg['message']}\" (Similarity: {msg['similarity']:.2f})")
            
            message_context = "\n".join(context_texts)
            system_prompt += f"\n\nFor additional context, here are some of your past messages that seem relevant to the current question:\n\n{message_context}\n\nUse these messages to inform your response in a natural way."
        
        return system_prompt

personality_service = PersonalityService() 