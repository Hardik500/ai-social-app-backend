import os
from google import genai

class GeminiProvider:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

    async def generate_chat(self, messages, system_prompt=None, stream=False, format_json=False):
        # Prepare the conversation history in the new SDK format
        history = []
        if system_prompt:
            history.append({"role": "system", "parts": [system_prompt]})
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history.append({"role": role, "parts": [content]})
        # The new SDK expects a flat list of message contents
        # We'll concatenate all user/assistant messages for context
        # But the API supports a list of messages as 'contents'
        import asyncio
        loop = asyncio.get_event_loop()
        def call():
            response = self.client.models.generate_content(
                model=self.chat_model,
                contents=history
            )
            return response
        response = await loop.run_in_executor(None, call)
        return {"message": {"content": response.text}}

    async def generate_embedding(self, text):
        import asyncio
        loop = asyncio.get_event_loop()
        def call():
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text
            )
            return response.embedding
        embedding = await loop.run_in_executor(None, call)
        return embedding 