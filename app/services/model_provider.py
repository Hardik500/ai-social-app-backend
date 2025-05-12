import os
from app.services.model_providers.ollama_provider import OllamaProvider
from app.services.model_providers.gemini_provider import GeminiProvider

class ModelProvider:
    def __init__(self):
        self.provider_name = os.getenv("CHAT_PROVIDER", "ollama").lower()
        print(f"Initializing ModelProvider with {self.provider_name}")
        
        if self.provider_name == "ollama":
            self.provider = OllamaProvider()
        elif self.provider_name == "gemini":
            self.provider = GeminiProvider()
        else:
            raise ValueError(f"Unknown CHAT_PROVIDER: {self.provider_name}")

    async def generate_chat(self, messages, system_prompt=None, stream=False, format_json=False, apply_rate_limit=False):        
        # Call the provider's generate_chat method and await the result
        result = await self.provider.generate_chat(messages, system_prompt, stream, format_json, apply_rate_limit)
        
        # Return the result to the caller
        return result

    async def generate_embedding(self, text, apply_rate_limit=False):
        return await self.provider.generate_embedding(text, apply_rate_limit)

# Initialize the model provider
model_provider = ModelProvider() 