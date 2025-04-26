import os
import httpx
import asyncio

class OllamaProvider:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.semaphore = asyncio.Semaphore(5)

    async def generate_chat(self, messages, system_prompt=None, stream=False, format_json=False):
        url = f"{self.base_url}/api/chat"
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages)
        payload = {
            "model": self.chat_model,
            "messages": chat_messages,
            "stream": stream
        }
        if format_json:
            payload["format"] = "json"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=180.0)
            response.raise_for_status()
            return response.json()

    async def generate_embedding(self, text):
        url = f"{self.base_url}/api/embeddings"
        async with self.semaphore:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"] 