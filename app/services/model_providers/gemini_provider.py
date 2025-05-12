import os
from google import genai
import asyncio
import json
import time

class GeminiProvider:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        self.last_api_call_time = 0
        self.delay_seconds = int(os.getenv("GEMINI_RATE_LIMIT_DELAY", 10))  # Default 10 seconds delay

    async def generate_chat(self, messages, system_prompt=None, stream=False, format_json=False, apply_rate_limit=False):
        # Add delay to avoid rate limiting, but only if apply_rate_limit is True
        if apply_rate_limit:
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call_time
            
            if time_since_last_call < self.delay_seconds:
                # Calculate how much longer we need to wait
                wait_time = self.delay_seconds - time_since_last_call
                print(f"Rate limit prevention: Waiting {wait_time:.2f} seconds before making API call")
                await asyncio.sleep(wait_time)
            
            # Update the last API call time
            self.last_api_call_time = time.time()
        
        # Gemini SDK expects a list of strings for chat history
        contents = []
        if system_prompt:
            contents.append(system_prompt)
        for msg in messages:
            # Only the message content is needed
            contents.append(msg.get("content", ""))
        
        if stream:
            # Create a streaming response object
            return GeminiStreamingResponse(
                contents=contents, 
                client=self.client,
                model_name=self.chat_model, 
                format_json=format_json,
                apply_rate_limit=apply_rate_limit
            )
        else:
            # For non-streaming, run the generation and return the result
            loop = asyncio.get_event_loop()
            def call():
                response = self.client.models.generate_content(
                    model=self.chat_model,
                    contents=contents
                )
                return response
            try:
                response = await loop.run_in_executor(None, call)
                text = response.text.strip()
                # If the model output is a JSON array string, parse it and return as list of messages
                if text.startswith("[") and text.endswith("]"):
                    try:
                        parsed = json.loads(text)
                        # If it's a list of dicts with 'content', return as is
                        if isinstance(parsed, list) and all(isinstance(m, dict) and 'content' in m for m in parsed):
                            return {"message": {"content": json.dumps(parsed)}}
                    except Exception:
                        pass
                # Otherwise, return as a single message
                return {"message": {"content": text}}
            except Exception as e:
                print(f"Exception when calling model provider: {str(e)}")
                # In case of error, still update the last call time to maintain rate limiting
                if apply_rate_limit:
                    self.last_api_call_time = time.time()
                raise

    async def generate_embedding(self, text, apply_rate_limit=False):
        # Add delay to avoid rate limiting for embeddings too, but only if apply_rate_limit is True
        if apply_rate_limit:
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call_time
            
            if time_since_last_call < self.delay_seconds:
                # Calculate how much longer we need to wait
                wait_time = self.delay_seconds - time_since_last_call
                print(f"Rate limit prevention: Waiting {wait_time:.2f} seconds before making embedding API call")
                await asyncio.sleep(wait_time)
            
            # Update the last API call time
            self.last_api_call_time = time.time()
        
        loop = asyncio.get_event_loop()
        def call():
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text
            )
            # Return the first embedding vector
            return response.embeddings[0].values
        try:
            embedding = await loop.run_in_executor(None, call)
            return embedding
        except Exception as e:
            print(f"Exception when calling embedding API: {str(e)}")
            # In case of error, still update the last call time to maintain rate limiting
            if apply_rate_limit:
                self.last_api_call_time = time.time()
            raise

class GeminiStreamingResponse:
    def __init__(self, contents, client, model_name, format_json, apply_rate_limit=False):
        self.contents = contents
        self.client = client
        self.model_name = model_name
        self.format_json = format_json
        self.stream = None
        self.response_iter = None
        self.last_api_call_time = 0
        self.delay_seconds = int(os.getenv("GEMINI_RATE_LIMIT_DELAY", 10))  # Default 10 seconds delay
        self.apply_rate_limit = apply_rate_limit
    
    async def __aenter__(self):
        # We use client's last API call time from the parent class
        # Get the current time
        current_time = time.time()
        
        # Check when the last API call was made from the provider
        # The provider already updated its last_api_call_time before creating this object
        # So we don't need to check any other timing variables
        
        loop = asyncio.get_event_loop()
        
        def start_stream():
            try:
                # Use the dedicated streaming API
                response = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=self.contents
                )
                # Convert the generator to a list of chunks
                # This is necessary because we can't pass a generator across threads
                chunks = list(response)
                return chunks
            except Exception as e:
                print(f"Error starting Gemini stream: {str(e)}")
                import traceback
                traceback.print_exc()
                raise e
        
        self.stream = await loop.run_in_executor(None, start_stream)
        self.response_iter = iter(self.stream)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Clean up resources
        self.stream = None
        self.response_iter = None
        return False  # Don't suppress exceptions
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.response_iter:
            raise StopAsyncIteration
        
        try:
            chunk = next(self.response_iter)
            
            # Extract the text from the chunk
            text = ""
            if hasattr(chunk, 'candidates') and len(chunk.candidates) > 0:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            text = part.text
            
            # Format as Ollama-compatible JSON for consistent interface
            result = {
                "message": {
                    "content": text
                }
            }
            
            return json.dumps(result)
        except StopIteration:
            raise StopAsyncIteration
        except Exception as e:
            print(f"Error in Gemini streaming: {str(e)}")
            import traceback
            traceback.print_exc()
            raise StopAsyncIteration 