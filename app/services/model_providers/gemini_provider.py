import os
from google import genai

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
        import asyncio
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
        import asyncio
        loop = asyncio.get_event_loop()
        def call():
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text
            )
            print("DEBUG: embed_content response:", response)
            print("DEBUG: dir(response):", dir(response))
            # Return the first embedding vector
            return response.embeddings[0].values
        embedding = await loop.run_in_executor(None, call)
        return embedding 