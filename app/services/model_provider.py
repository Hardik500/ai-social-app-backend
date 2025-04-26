import os
from app.services.model_providers.ollama_provider import OllamaProvider
from app.services.model_providers.gemini_provider import GeminiProvider

class ModelProvider:
    def __init__(self):
        self.provider_name = os.getenv("CHAT_PROVIDER", "ollama").lower()
        if self.provider_name == "ollama":
            self.provider = OllamaProvider()
        elif self.provider_name == "gemini":
            self.provider = GeminiProvider()
        else:
            raise ValueError(f"Unknown CHAT_PROVIDER: {self.provider_name}")

    async def generate_chat(self, messages, system_prompt=None, stream=False, format_json=False):
        return await self.provider.generate_chat(messages, system_prompt, stream, format_json)

    async def generate_embedding(self, text):
        return await self.provider.generate_embedding(text)

model_provider = ModelProvider() 