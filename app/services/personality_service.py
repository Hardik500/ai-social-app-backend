import os
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
import time
import hashlib
import redis
from sqlalchemy.orm import Session
from sqlalchemy import asc
from app.models.user import User
from app.models.conversation import Message, ConversationHistory
from app.models.personality import PersonalityProfile
from app.services.embedding_service import embedding_service
from app.core.prompt_manager import prompt_manager
from app.services.model_provider import model_provider
import re

class PersonalityService:
    def __init__(self):
        self.cache_size = 100
        redis_url = os.getenv("REDIS_URL")
        self.redis_client = None
        self.cache_expiry = 3600
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                print(f"Redis cache initialized at {redis_url}")
            except Exception as e:
                print(f"Failed to connect to Redis: {str(e)}")
                self.redis_client = None
        self.response_cache = {}

    async def generate_profile(self, user_id: int, db: Session) -> Optional[PersonalityProfile]:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            print(f"User with ID {user_id} not found")
            return None
        messages = db.query(Message).filter(Message.user_id == user_id).all()
        message_count = len(messages)
        if not messages or message_count < 5:
            print(f"Insufficient messages for user {user.username}: {message_count} (Need at least 5)")
            return None
        existing_profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id, 
            PersonalityProfile.is_active == True
        ).first()
        if existing_profile and hasattr(existing_profile, 'last_message_id') and existing_profile.last_message_id:
            try:
                new_messages = db.query(Message).filter(
                    Message.user_id == user_id,
                    Message.id > existing_profile.last_message_id
                ).all()
                if len(new_messages) >= 5:
                    return await self.update_profile_incrementally(
                        user_id, existing_profile, new_messages, db
                    )
            except Exception as e:
                print(f"Error during incremental update check: {str(e)}")
                pass
        MAX_MESSAGES = 1000
        if message_count > MAX_MESSAGES:
            selected_messages = self._select_representative_messages(messages, MAX_MESSAGES, db)
            messages = selected_messages
        message_texts = [msg.content for msg in messages]
        system_prompt = prompt_manager.get_template("personality_analysis")
        traits_and_description = await self._generate_analysis(message_texts, system_prompt)
        if traits_and_description is None:
            print(f"Failed to generate personality analysis for user {user.username} (ID: {user_id})")
            return None
        traits = traits_and_description.get("traits", {})
        communication_style = traits_and_description.get("communication_style", {})
        interests = traits_and_description.get("interests", [])
        values = traits_and_description.get("values", [])
        summary = traits_and_description.get("summary", "")
        response_length = traits_and_description.get("response_length", "moderate")
        common_phrases = traits_and_description.get("common_phrases", [])
        emotional_responses = traits_and_description.get("emotional_responses", "")
        conflict_style = traits_and_description.get("conflict_style", "")
        description = prompt_manager.format_template(
            "description_template",
            summary=summary,
            openness=traits.get('openness', 'N/A'),
            conscientiousness=traits.get('conscientiousness', 'N/A'),
            extraversion=traits.get('extraversion', 'N/A'),
            agreeableness=traits.get('agreeableness', 'N/A'),
            neuroticism=traits.get('neuroticism', 'N/A'),
            communication_style=communication_style if isinstance(communication_style, str) else \
                ', '.join([f'{k}: {v}' for k, v in communication_style.items()]) \
                if isinstance(communication_style, dict) else communication_style,
            interests=interests if isinstance(interests, str) else \
                ', '.join(interests) if isinstance(interests, list) else interests,
            values=values if isinstance(values, str) else \
                ', '.join(values) if isinstance(values, list) else values
        )
        system_prompt = prompt_manager.format_template(
            "personality_simulation",
            username=user.username,
            openness=traits.get('openness', 5),
            conscientiousness=traits.get('conscientiousness', 5),
            extraversion=traits.get('extraversion', 5),
            agreeableness=traits.get('agreeableness', 5),
            neuroticism=traits.get('neuroticism', 5),
            communication_style=communication_style if isinstance(communication_style, str) else \
                ', '.join([f'{k}: {v}' for k, v in communication_style.items()]) \
                if isinstance(communication_style, dict) else communication_style,
            interests=interests if isinstance(interests, str) else \
                ', '.join(interests) if isinstance(interests, list) else interests,
            values=values if isinstance(values, str) else \
                ', '.join(values) if isinstance(values, list) else values,
            summary=summary,
            response_length=response_length,
            common_phrases=common_phrases if isinstance(common_phrases, str) else ', '.join(common_phrases) if isinstance(common_phrases, list) else common_phrases,
            emotional_responses=emotional_responses,
            conflict_style=conflict_style,
            topic="general conversation",
            participants=f"{user.username} and others",
            mood="neutral"
        )
        embedding_task = asyncio.create_task(embedding_service.generate_embedding(description))
        embedding = await embedding_task
        if existing_profile:
            # Update the existing profile in place
            existing_profile.traits = traits_and_description
            existing_profile.description = description
            existing_profile.embedding = embedding
            existing_profile.message_count = message_count
            existing_profile.system_prompt = system_prompt
            existing_profile.is_active = True
            existing_profile.last_message_id = max(msg.id for msg in messages) if messages else None
            db.commit()
            db.refresh(existing_profile)
            return existing_profile
        else:
            # If no existing profile, create a new one
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
            db.add(new_profile)
            db.commit()
            db.refresh(new_profile)
            return new_profile

    async def _generate_analysis(self, messages: List[str], system_prompt: str) -> Optional[Dict[str, Any]]:
        max_messages_per_request = 50
        if len(messages) > max_messages_per_request:
            message_chunks = [messages[i:i + max_messages_per_request] \
                             for i in range(0, len(messages), max_messages_per_request)]
            chunk_results = []
            for chunk in message_chunks:
                result = await self._generate_chunk_analysis(chunk, system_prompt, is_chunk=True)
                if result:
                    chunk_results.append(result)
            if not chunk_results:
                return None
            merged_result = self._merge_chunk_results(chunk_results)
            return merged_result
        return await self._generate_chunk_analysis(messages, system_prompt, is_chunk=False)

    async def _generate_chunk_analysis(self, messages: List[str], system_prompt: str, is_chunk: bool) -> Optional[Dict[str, Any]]:
        processed_messages = messages
        message_text = "\n---\n".join(processed_messages)
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
            result = await model_provider.generate_chat(chat_messages, system_prompt=None, stream=False, format_json=True)
            if "message" in result and "content" in result["message"]:
                content = result["message"]["content"]
                try:
                    print(f"Raw model response content: {content}")  # Debug print
                    content = self.extract_json_from_response(content)
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, dict):
                        print(f"Parsed content is not a dict: {parsed_content}")
                        return None
                    return parsed_content
                except Exception as e:
                    print(f"Failed to parse JSON from response: {content}")
                    print(f"JSON Error: {str(e)}")
                    return None
            else:
                print(f"Unexpected API response format: {result}")
                return None
        except Exception as e:
            print(f"Exception when calling model provider: {str(e)}")
            return None

    def extract_json_from_response(self, content: str) -> str:
        """
        Extract the first valid JSON object from a string, removing code fences and extra text.
        """
        content = content.strip()
        # Remove ```json or ``` at the start
        content = re.sub(r"^```json", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"^```", "", content).strip()
        # Remove trailing ```
        content = re.sub(r"```$", "", content).strip()
        # Find the first { and last } to extract the JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start:end+1]
        return content

    async def generate_response(self, user_id: int, question: str, db: Session, log_history: bool = True, multi_message: bool = False) -> Optional[List[Dict[str, str]]]:
        cache_key = self._get_cache_key(user_id, question)
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            try:
                # Try to parse the cached response as JSON for multi-message format
                return json.loads(cached_response)
            except:
                # Fall back to single message format if not JSON
                return [{"content": cached_response, "type": "text"}]
                
        # Store the current question in ConversationHistory as a 'user' message
        if log_history:
            print(f"Logging history for user {user_id} and question: {question[:50]}...")
            user_history_entry = ConversationHistory(user_id=user_id, role='user', content=question)
            db.add(user_history_entry)
            db.commit()
        else:
            print(f"Skipping history logging for user {user_id} and question: {question[:50]}...")
            
        # Fetch recent ingested messages
        recent_messages = (
            db.query(Message)
            .filter(Message.user_id == user_id)
            .order_by(Message.created_at.desc())
            .limit(500)
            .all()
        )
        
        # Fetch current session's conversation history
        session_history = db.query(ConversationHistory).filter(ConversationHistory.user_id == user_id).order_by(asc(ConversationHistory.created_at)).all()
        
        user_context = f"\nUser Information:\n"
        user = db.query(User).filter(User.id == user_id).first()
        profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id,
            PersonalityProfile.is_active == True
        ).first()
        
        if not user or not profile:
            return None
            
        user_context += f"- Username: {user.username}\n"
        if hasattr(user, 'email') and user.email:
            user_context += f"- Email: {user.email}\n"
        if hasattr(user, 'full_name') and user.full_name:
            user_context += f"- Full Name: {user.full_name}\n"
        if hasattr(user, 'created_at') and user.created_at:
            user_context += f"- Member since: {user.created_at.strftime('%Y-%m-%d')}\n"
        if hasattr(profile, 'message_count') and profile.message_count:
            user_context += f"- Total messages: {profile.message_count}\n"
            
        conversation_history = ""
        if recent_messages:
            conversation_history = "\nRecent ingested conversation history:\n"
            for msg in reversed(recent_messages):
                timestamp = f" ({msg.created_at.strftime('%Y-%m-%d %H:%M')})" if hasattr(msg, 'created_at') and msg.created_at else ""
                conversation_history += f"- {msg.content}{timestamp}\n"
                
        session_history_str = ""
        if session_history:
            session_history_str = "\nCurrent session conversation history:\n"
            for entry in session_history:
                role = "You" if entry.role == "user" else "AI"
                session_history_str += f"- {role}: {entry.content}\n"
                
        # Explicitly include the current question in the prompt context
        current_question_context = f"\nCurrent question: {question}\n"
        
        system_prompt = self._enhance_system_prompt_for_question(
            profile.system_prompt + user_context + conversation_history + session_history_str + current_question_context,
            user.username,
            question
        )
        
        # Add multi-message instruction if enabled
        if multi_message:
            system_prompt += "\n\nIMPORTANT: If appropriate for this conversation, respond with multiple sequential messages instead of one long message. This creates a more natural conversation flow."
        
        print(f"Enhanced system prompt: {system_prompt}...")
        
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            # Only use format_json when multi_message is True
            result = await model_provider.generate_chat(chat_messages, system_prompt=None, stream=False, format_json=multi_message)
            
            if "message" in result and "content" in result["message"]:
                response_content = result["message"]["content"]
                
                # If the response looks like a JSON array, parse it and return as a list of messages
                if response_content.strip().startswith("[") and response_content.strip().endswith("]"):
                    try:
                        response_messages = json.loads(response_content)
                        if isinstance(response_messages, list):
                            for message in response_messages:
                                if "type" not in message:
                                    message["type"] = "text"
                            # Store each message in ConversationHistory
                            if log_history:
                                for message in response_messages:
                                    ai_history_entry = ConversationHistory(user_id=user_id, role='ai', content=message["content"])
                                    db.add(ai_history_entry)
                                db.commit()
                            self._add_to_cache(self._get_cache_key(user_id, question), response_content)
                            return response_messages
                    except Exception:
                        pass
                
                # Otherwise, treat as a single message
                single_message = [{"content": response_content, "type": "text"}]
                # Store the AI's answer in ConversationHistory
                if log_history:
                    ai_history_entry = ConversationHistory(user_id=user_id, role='ai', content=response_content)
                    db.add(ai_history_entry)
                    db.commit()
                self._add_to_cache(self._get_cache_key(user_id, question), json.dumps(single_message))
                return single_message
                    
        except Exception as e:
            print(f"Exception when calling model provider for user {user.username}: {str(e)}")
            return None

    def _enhance_system_prompt_for_question(self, system_prompt: str, username: str, question: str) -> str:
        topic = "general conversation"
        mood = "neutral"
        question_lower = question.lower()
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
        if any(word in question_lower for word in ["urgent", "emergency", "critical", "asap", "quickly"]):
            mood = "urgent"
        elif any(word in question_lower for word in ["sad", "sorry", "upset", "disappointed"]):
            mood = "empathetic"
        elif any(word in question_lower for word in ["excited", "happy", "great", "awesome"]):
            mood = "enthusiastic"
        elif any(word in question_lower for word in ["confused", "don't understand", "unclear"]):
            mood = "clarifying"
        updated_prompt = system_prompt
        try:
            if "Current conversation context:" in system_prompt:
                topic_pattern = r"- Topic: .*"
                if re.search(topic_pattern, system_prompt):
                    updated_prompt = re.sub(topic_pattern, f"- Topic: {topic}", updated_prompt)
                participants_pattern = r"- Participants: .*"
                if re.search(participants_pattern, system_prompt):
                    updated_prompt = re.sub(participants_pattern, f"- Participants: {username} and the person asking", updated_prompt)
                mood_pattern = r"- Mood: .*"
                if re.search(mood_pattern, system_prompt):
                    updated_prompt = re.sub(mood_pattern, f"- Mood: {mood}", updated_prompt)
            if not updated_prompt.endswith("\n"):
                updated_prompt += "\n"
            updated_prompt += f"\nCurrent question context: You are now responding to a question about '{topic}' with a '{mood}' tone."
        except Exception as e:
            print(f"Error enhancing system prompt: {str(e)}")
            return system_prompt
        return updated_prompt

    def _get_cache_key(self, user_id: int, question: str) -> str:
        key_string = f"{user_id}:{question}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"personality_response:{cache_key}")
                if cached:
                    return cached.decode('utf-8')
            except Exception as e:
                print(f"Redis cache error: {str(e)}")
        return self.response_cache.get(cache_key)

    def _add_to_cache(self, cache_key: str, response: str) -> None:
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"personality_response:{cache_key}", 
                    self.cache_expiry,
                    response
                )
            except Exception as e:
                print(f"Redis cache error: {str(e)}")
        self.response_cache[cache_key] = response
        if len(self.response_cache) > self.cache_size:
            keys_to_remove = list(self.response_cache.keys())[:-self.cache_size]
            for key in keys_to_remove:
                del self.response_cache[key]

    async def preload_common_questions(self, user_id: int, common_questions: List[str], db: Session) -> None:
        for question in common_questions:
            asyncio.create_task(self.generate_response(user_id, question, db))

    async def preload_related_questions(self, user_id: int, original_question: str, original_answer: str, db: Session) -> None:
        profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id,
            PersonalityProfile.is_active == True
        ).first()
        if not profile:
            return
        follow_up_questions = await self._generate_related_questions(
            original_question, 
            original_answer, 
            profile.system_prompt
        )
        for question in follow_up_questions:
            asyncio.create_task(self.generate_response(user_id, question, db, log_history=False))

    async def _generate_related_questions(self, original_question: str, original_answer: str, system_prompt: str) -> List[str]:
        followup_system_prompt = "You are a helpful assistant tasked with generating natural follow-up questions. " + \
                               "Based on the conversation, suggest questions that would naturally continue the discussion " + \
                               "in a way that's consistent with the personality profile. Focus on questions that are " + \
                               "relevant to the previous exchange and would elicit informative or interesting responses."
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
            response = await model_provider.generate_chat(chat_messages, system_prompt=None, stream=False, format_json=True)
            if "message" in response and "content" in response["message"]:
                content = response["message"]["content"]
                try:
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    elif content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    questions = json.loads(content.strip())
                    if isinstance(questions, list) and len(questions) > 0:
                        return questions[:3]
                except json.JSONDecodeError:
                    matches = re.findall(r'"([^"]+\?)"', content)
                    if matches:
                        return matches[:3]
            return []
        except Exception as e:
            print(f"Error generating related questions: {str(e)}")
            return []

    async def generate_response_stream(self, user_id: int, question: str, db: Session, multi_message: bool = False):
        cache_key = self._get_cache_key(user_id, question)
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            print(f"Cache hit for user {user_id} and question: {question[:30]}...")
            try:
                # Try to parse cached response as JSON for multi-message format
                messages = json.loads(cached_response)
                yield json.dumps({
                    "type": "cached",
                    "content": messages,
                    "done": True
                }) + "\n"
            except:
                # Fall back to single message format
                yield json.dumps({
                    "type": "cached",
                    "content": [{"content": cached_response, "type": "text"}],
                    "done": True
                }) + "\n"
            return
            
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
            
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            yield json.dumps({
                "error": "User not found",
                "done": True
            }) + "\n"
            return
            
        # Log user message in history
        user_history_entry = ConversationHistory(user_id=user_id, role='user', content=question)
        db.add(user_history_entry)
        db.commit()
        
        # Build context the same way as in generate_response
        recent_messages = (
            db.query(Message)
            .filter(Message.user_id == user_id)
            .order_by(Message.created_at.desc())
            .limit(50)
            .all()
        )
        
        # Fetch current session's conversation history
        session_history = db.query(ConversationHistory).filter(
            ConversationHistory.user_id == user_id
        ).order_by(asc(ConversationHistory.created_at)).all()
        
        user_context = f"\nUser Information:\n"
        user_context += f"- Username: {user.username}\n"
        if hasattr(user, 'email') and user.email:
            user_context += f"- Email: {user.email}\n"
        if hasattr(user, 'full_name') and user.full_name:
            user_context += f"- Full Name: {user.full_name}\n"
        
        conversation_history = ""
        if recent_messages:
            conversation_history = "\nRecent conversation history:\n"
            for msg in reversed(recent_messages):
                conversation_history += f"- {msg.content}\n"
                
        session_history_str = ""
        if session_history:
            session_history_str = "\nCurrent session conversation history:\n"
            for entry in session_history:
                role = "User" if entry.role == "user" else "AI"
                session_history_str += f"- {role}: {entry.content}\n"
                
        system_prompt = self._enhance_system_prompt_for_question(
            profile.system_prompt + user_context + conversation_history + session_history_str, 
            user.username, 
            question
        )
        
        # Add multi-message instruction if enabled
        if multi_message:
            system_prompt += "\n\nIMPORTANT: If appropriate for this conversation, respond with multiple sequential messages instead of one long message. This creates a more natural conversation flow."
        
        print(f"Enhanced system prompt: {system_prompt}...")
        
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            raw_response = ""
            current_message = {"content": "", "type": "text"} 
            messages_stack = []
            is_json_array = False
            json_collection_mode = False
            
            # Get streaming response object
            streaming_obj = await model_provider.generate_chat(chat_messages, system_prompt=None, stream=True, format_json=True)
            
            # Use async with on the streaming object
            async with streaming_obj as response:
                async for line in response:
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                if content:
                                    raw_response += content
                                    
                                    # Check if we're starting JSON array collection
                                    if content.strip().startswith("[") and not json_collection_mode:
                                        json_collection_mode = True
                                        is_json_array = True
                                    
                                    if json_collection_mode:
                                        # Just accumulate for later parsing
                                        yield json.dumps({
                                            "type": "token",
                                            "content": content,
                                            "done": False
                                        }) + "\n"
                                    else:
                                        # Normal text streaming - accumulate in current_message
                                        current_message["content"] += content
                                        yield json.dumps({
                                            "type": "token",
                                            "content": content,
                                            "done": False
                                        }) + "\n"
                        except json.JSONDecodeError:
                            pass
            
            # Process the final response
            if json_collection_mode:
                try:
                    # Try to parse the entire response as a JSON array
                    messages = json.loads(raw_response)
                    if isinstance(messages, list):
                        # Add default type if missing
                        for message in messages:
                            if "type" not in message:
                                message["type"] = "text"
                        
                        # Store each message in conversation history
                        for message in messages:
                            ai_history_entry = ConversationHistory(
                                user_id=user_id, 
                                role='ai', 
                                content=message["content"]
                            )
                            db.add(ai_history_entry)
                        db.commit()
                        
                        # Cache and return the parsed messages
                        self._add_to_cache(cache_key, raw_response)
                        yield json.dumps({
                            "type": "complete",
                            "content": messages,
                            "done": True
                        }) + "\n"
                        return
                except:
                    # If JSON parsing fails, fall back to treating it as plain text
                    json_collection_mode = False
                    
            if not json_collection_mode:
                # Single message mode - finalize the message we've been building
                if raw_response:
                    # Create a single message
                    message = {"content": raw_response, "type": "text"}
                    messages = [message]
                    
                    # Store in conversation history
                    ai_history_entry = ConversationHistory(
                        user_id=user_id, 
                        role='ai', 
                        content=raw_response
                    )
                    db.add(ai_history_entry)
                    db.commit()
                    
                    # Cache the message as JSON array
                    self._add_to_cache(cache_key, json.dumps(messages))
                    yield json.dumps({
                        "type": "complete",
                        "content": messages,
                        "done": True
                    }) + "\n"
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
        MAX_DELTA_MESSAGES = 50
        if len(new_messages) > MAX_DELTA_MESSAGES:
            selected_messages = self._select_representative_messages(new_messages, MAX_DELTA_MESSAGES, db)
            print(f"Selected {len(selected_messages)} representative new messages from {len(new_messages)} total new")
            new_messages = selected_messages
        message_texts = [msg.content for msg in new_messages]
        existing_traits = existing_profile.traits
        system_prompt = prompt_manager.get_template("personality_incremental_update")
        formatted_prompt = system_prompt.format(
            existing_profile=json.dumps(existing_traits, indent=2),
            message_count=len(new_messages)
        )
        traits_delta = await self._generate_analysis(message_texts, formatted_prompt)
        # Defensive: If traits_delta is not a dict, try to extract JSON and parse again
        if not isinstance(traits_delta, dict):
            print(f"traits_delta is not a dict: {traits_delta}")
            # Try to extract JSON if traits_delta is a string
            if isinstance(traits_delta, str):
                from .personality_service import PersonalityService  # Avoid circular import
                try:
                    json_str = self.extract_json_from_response(traits_delta)
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        traits_delta = parsed
                    else:
                        print(f"Parsed content is not a dict: {parsed}")
                        print(f"Failed to generate incremental analysis, falling back to full profile generation")
                        existing_profile.is_active = False
                        db.commit()
                        return await self.generate_profile(user_id, db)
                except Exception as e:
                    print(f"Error during defensive JSON extraction: {e}")
                    print(f"Raw response: {traits_delta}")
                    print(f"Failed to generate incremental analysis, falling back to full profile generation")
                    existing_profile.is_active = False
                    db.commit()
                    return await self.generate_profile(user_id, db)
            else:
                print(f"Failed to generate incremental analysis, falling back to full profile generation")
                existing_profile.is_active = False
                db.commit()
                return await self.generate_profile(user_id, db)
        updated_traits = self._merge_traits(existing_traits, traits_delta)
        latest_message_id = max(msg.id for msg in new_messages) if new_messages else existing_profile.last_message_id
        total_message_count = existing_profile.message_count + len(new_messages)
        summary = updated_traits.get("summary", "")
        response_length = updated_traits.get("response_length", "moderate")
        common_phrases = updated_traits.get("common_phrases", [])
        emotional_responses = updated_traits.get("emotional_responses", "")
        conflict_style = updated_traits.get("conflict_style", "")
        traits = updated_traits.get("traits", {})
        communication_style = updated_traits.get("communication_style", {})
        interests = updated_traits.get("interests", [])
        values = updated_traits.get("values", [])
        description = prompt_manager.format_template(
            "description_template",
            summary=summary,
            openness=traits.get('openness', 'N/A'),
            conscientiousness=traits.get('conscientiousness', 'N/A'),
            extraversion=traits.get('extraversion', 'N/A'),
            agreeableness=traits.get('agreeableness', 'N/A'),
            neuroticism=traits.get('neuroticism', 'N/A'),
            communication_style=communication_style if isinstance(communication_style, str) else \
                ', '.join([f'{k}: {v}' for k, v in communication_style.items()]) \
                if isinstance(communication_style, dict) else communication_style,
            interests=interests if isinstance(interests, str) else \
                ', '.join(interests) if isinstance(interests, list) else interests,
            values=values if isinstance(values, str) else \
                ', '.join(values) if isinstance(values, list) else values
        )
        username = db.query(User).filter(User.id == user_id).first().username
        system_prompt = prompt_manager.format_template(
            "personality_simulation",
            username=username,
            openness=traits.get('openness', 5),
            conscientiousness=traits.get('conscientiousness', 5),
            extraversion=traits.get('extraversion', 5),
            agreeableness=traits.get('agreeableness', 5),
            neuroticism=traits.get('neuroticism', 5),
            communication_style=communication_style if isinstance(communication_style, str) else \
                ', '.join([f'{k}: {v}' for k, v in communication_style.items()]) \
                if isinstance(communication_style, dict) else communication_style,
            interests=interests if isinstance(interests, str) else \
                ', '.join(interests) if isinstance(interests, list) else interests,
            values=values if isinstance(values, str) else \
                ', '.join(values) if isinstance(values, list) else values,
            summary=summary,
            response_length=response_length,
            common_phrases=common_phrases if isinstance(common_phrases, str) else ', '.join(common_phrases) if isinstance(common_phrases, list) else common_phrases,
            emotional_responses=emotional_responses,
            conflict_style=conflict_style,
            topic="general conversation",
            participants=f"{username} and others",
            mood="neutral"
        )
        embedding = await embedding_service.generate_embedding(description)
        change_log = {
            "timestamp": time.time(),
            "new_message_count": len(new_messages),
            "total_message_count": total_message_count,
            "changes": traits_delta.get("changes", {})
        }
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
        delta_embedding = None
        try:
            delta_text = traits_delta.get("delta_summary", "")
            if delta_text:
                delta_embedding = await embedding_service.generate_embedding(delta_text)
        except:
            delta_embedding = None
        existing_profile.traits = updated_traits
        existing_profile.description = description
        existing_profile.embedding = embedding
        existing_profile.message_count = total_message_count
        existing_profile.system_prompt = system_prompt
        existing_profile.last_message_id = latest_message_id
        if hasattr(existing_profile, 'delta_embeddings') and delta_embedding:
            if existing_profile.delta_embeddings:
                if isinstance(existing_profile.delta_embeddings, str):
                    try:
                        existing_delta_embeddings = json.loads(existing_profile.delta_embeddings)
                    except:
                        existing_delta_embeddings = []
                else:
                    existing_delta_embeddings = existing_profile.delta_embeddings
                existing_delta_embeddings.append(delta_embedding)
                existing_profile.delta_embeddings = existing_delta_embeddings
            else:
                existing_profile.delta_embeddings = [delta_embedding]
        if hasattr(existing_profile, 'change_log'):
            existing_profile.change_log = change_log_json
        db.commit()
        db.refresh(existing_profile)
        return existing_profile

    def _merge_traits(self, existing_traits: Dict[str, Any], delta_traits: Dict[str, Any]) -> Dict[str, Any]:
        updated_traits = json.loads(json.dumps(existing_traits))
        if "traits" in delta_traits and "traits" in updated_traits:
            for trait, value in delta_traits["traits"].items():
                updated_traits["traits"][trait] = value
        if "communication_style" in delta_traits:
            if isinstance(delta_traits["communication_style"], dict) and isinstance(updated_traits.get("communication_style", {}), dict):
                for key, value in delta_traits["communication_style"].items():
                    updated_traits.setdefault("communication_style", {})[key] = value
            else:
                updated_traits["communication_style"] = delta_traits["communication_style"]
        if "interests" in delta_traits:
            if isinstance(delta_traits["interests"], list) and isinstance(updated_traits.get("interests", []), list):
                seen = set()
                combined = []
                for item in delta_traits["interests"] + updated_traits.get("interests", []):
                    if item.lower() not in seen:
                        combined.append(item)
                        seen.add(item.lower())
                updated_traits["interests"] = combined[:15]
            else:
                updated_traits["interests"] = delta_traits["interests"]
        if "values" in delta_traits:
            if isinstance(delta_traits["values"], list) and isinstance(updated_traits.get("values", []), list):
                seen = set()
                combined = []
                for item in delta_traits["values"] + updated_traits.get("values", []):
                    if item.lower() not in seen:
                        combined.append(item)
                        seen.add(item.lower())
                updated_traits["values"] = combined[:10]
            else:
                updated_traits["values"] = delta_traits["values"]
        if "summary" in delta_traits:
            updated_traits["summary"] = delta_traits["summary"]
        return updated_traits

    def _merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunk_results:
            return {}
        merged = json.loads(json.dumps(chunk_results[0]))
        trait_counts = {}
        trait_mappings = {
            'openness': ['openness', 'openness_to_experience'],
            'conscientiousness': ['conscientiousness'],
            'extraversion': ['extraversion', 'extroversion'],
            'agreeableness': ['agreeableness'],
            'neuroticism': ['neuroticism']
        }
        for canonical_name in trait_mappings.keys():
            trait_counts[canonical_name] = []
        all_interests = set()
        all_values = set()
        communication_aspects = {}
        for chunk in chunk_results:
            if "traits" in chunk:
                for trait_name, value in chunk["traits"].items():
                    if isinstance(value, (int, float)) and 1 <= value <= 10:
                        canonical_name = None
                        for canon, variants in trait_mappings.items():
                            if trait_name.lower() in variants or trait_name.lower() == canon:
                                canonical_name = canon
                                break
                        if canonical_name:
                            trait_counts[canonical_name].append(value)
            if "interests" in chunk and isinstance(chunk["interests"], list):
                for interest in chunk["interests"]:
                    if isinstance(interest, str):
                        all_interests.add(interest)
                    elif isinstance(interest, dict) and "name" in interest:
                        all_interests.add(interest["name"])
            if "values" in chunk and isinstance(chunk["values"], list):
                for value in chunk["values"]:
                    if isinstance(value, str):
                        all_values.add(value)
                    elif isinstance(value, dict) and "name" in value:
                        all_values.add(value["name"])
            if "communication_style" in chunk and isinstance(chunk["communication_style"], dict):
                for aspect, value in chunk["communication_style"].items():
                    if aspect not in communication_aspects:
                        communication_aspects[aspect] = []
                    communication_aspects[aspect].append(value)
        if "traits" not in merged:
            merged["traits"] = {}
        for trait_name, values in trait_counts.items():
            if values:
                merged["traits"][trait_name] = round(sum(values) / len(values))
        merged["interests"] = list(all_interests)[:15]
        merged["values"] = list(all_values)[:10]
        if "communication_style" not in merged:
            merged["communication_style"] = {}
        for aspect, values in communication_aspects.items():
            if isinstance(values[0], (int, float)):
                merged["communication_style"][aspect] = sum(values) / len(values)
            else:
                from collections import Counter
                counter = Counter(values)
                merged["communication_style"][aspect] = counter.most_common(1)[0][0]
        if "summary" not in merged:
            merged["summary"] = ""
        summaries = [chunk.get("summary", "") for chunk in chunk_results if "summary" in chunk]
        if summaries:
            merged["summary"] = "This individual " + " ".join(summaries)
        return merged

    def _generate_plausible_question(self, message_content: str) -> str:
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if any(message_content.lower().startswith(greeting) for greeting in greetings):
            return "How are you doing today?"
        if "thanks" in message_content.lower() or "thank you" in message_content.lower():
            return "Could you help me with this issue?"
        if "because" in message_content.lower() or "since" in message_content.lower():
            return "Can you explain why this is happening?"
        opinion_words = ["think", "believe", "opinion", "feel", "view"]
        if any(word in message_content.lower() for word in opinion_words):
            return "What do you think about this?"
        return "What's your perspective on this situation?"

    async def build_rag_enhanced_system_prompt(
        self, user_id: int, username: str, relevant_messages: List[Dict], question: str, db: Session
    ) -> str:
        profile = db.query(PersonalityProfile).filter(
            PersonalityProfile.user_id == user_id,
            PersonalityProfile.is_active == True
        ).first()
        if not profile or not profile.traits:
            return f"You are roleplaying as {username}. Answer the following question as if you were this person."
        traits = profile.traits.get("traits", {})
        communication_style = profile.traits.get("communication_style", {})
        interests = profile.traits.get("interests", [])
        values = profile.traits.get("values", [])
        summary = profile.traits.get("summary", "")
        response_length = profile.traits.get("response_length", "moderate")
        common_phrases = profile.traits.get("common_phrases", [])
        emotional_responses = profile.traits.get("emotional_responses", "")
        conflict_style = profile.traits.get("conflict_style", "")
        example_messages = {}
        example_messages = {
            "question1": "How are you today?",
            "response1": "I'm doing well, thanks for asking!",
            "question2": "What do you think about this project?",
            "response2": "I think it's interesting and has a lot of potential."
        }
        if relevant_messages and len(relevant_messages) >= 2:
            example_messages["response1"] = relevant_messages[0]["message"]
            example_messages["question1"] = self._generate_plausible_question(relevant_messages[0]["message"])
            example_messages["response2"] = relevant_messages[1]["message"]
            example_messages["question2"] = self._generate_plausible_question(relevant_messages[1]["message"])
        system_prompt = prompt_manager.format_template(
            "personality_simulation",
            username=username,
            openness=traits.get('openness', 5),
            conscientiousness=traits.get('conscientiousness', 5),
            extraversion=traits.get('extraversion', 5),
            agreeableness=traits.get('agreeableness', 5),
            neuroticism=traits.get('neuroticism', 5),
            communication_style=communication_style if isinstance(communication_style, str) else \
                ', '.join([f'{k}: {v}' for k, v in communication_style.items()]) \
                if isinstance(communication_style, dict) else communication_style,
            interests=interests if isinstance(interests, str) else \
                ', '.join(interests) if isinstance(interests, list) else interests,
            values=values if isinstance(values, str) else \
                ', '.join(values) if isinstance(values, list) else values,
            summary=summary,
            response_length=response_length,
            common_phrases=common_phrases if isinstance(common_phrases, str) else ', '.join(common_phrases) if isinstance(common_phrases, list) else common_phrases,
            emotional_responses=emotional_responses,
            conflict_style=conflict_style,
            topic=f"answering '{question}'",
            participants=f"{username} and the person asking the question",
            mood="helpful"
        )
        if relevant_messages:
            context_texts = []
            for i, msg in enumerate(relevant_messages):
                context_texts.append(f"Message {i+1}: \"{msg['message']}\" (Similarity: {msg['similarity']:.2f})")
            message_context = "\n".join(context_texts)
            system_prompt += f"\n\nFor additional context, here are some of your past messages that seem relevant to the current question:\n\n{message_context}\n\nUse these messages to inform your response in a natural way."
        return system_prompt

    def _select_representative_messages(self, messages, max_count, db):
        """
        Select up to max_count representative messages from the list.
        This implementation simply picks the most recent messages.
        """
        # Sort messages by created_at if available, else by id
        if hasattr(messages[0], 'created_at'):
            sorted_msgs = sorted(messages, key=lambda m: m.created_at, reverse=True)
        else:
            sorted_msgs = sorted(messages, key=lambda m: m.id, reverse=True)
        return sorted_msgs[:max_count]

personality_service = PersonalityService() 