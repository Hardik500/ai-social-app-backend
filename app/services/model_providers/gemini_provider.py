import os
from google import genai
import asyncio
import json

class GeminiProvider:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

    async def generate_chat(self, messages, system_prompt=None, stream=False, format_json=False):
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
                format_json=format_json
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
            response = await loop.run_in_executor(None, call)
            return {"message": {"content": response.text}}

    async def generate_embedding(self, text):
        loop = asyncio.get_event_loop()
        def call():
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text
            )
            # Return the first embedding vector
            return response.embeddings[0].values
        embedding = await loop.run_in_executor(None, call)
        return embedding

class GeminiStreamingResponse:
    def __init__(self, contents, client, model_name, format_json):
        self.contents = contents
        self.client = client
        self.model_name = model_name
        self.format_json = format_json
        self.stream = None
        self.response_iter = None
    
    async def __aenter__(self):
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