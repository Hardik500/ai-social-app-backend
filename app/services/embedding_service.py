import httpx
import os
from dotenv import load_dotenv
import json
from sqlalchemy import text
from sqlalchemy.orm import Session

load_dotenv()

class EmbeddingService:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = "nomic-embed-text"  # Default embedding model
    
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embeddings for the given text using Ollama API directly."""
        url = f"{self.base_url}/api/embeddings"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={
                        "model": self.model,
                        "prompt": text
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["embedding"]
                else:
                    error_msg = f"Error generating embedding: {response.status_code} - {response.text}"
                    print(error_msg)
                    raise Exception(error_msg)
        except Exception as e:
            print(f"Exception when calling Ollama API: {str(e)}")
            raise e
    
    def schedule_embedding_generation(self, db: Session, text: str, table: str, id_column: str, id_value: int, text_column: str, embedding_column: str):
        """
        Schedule embedding generation using pgAI's vectorize_job function.
        This uses the vectorizer-worker to asynchronously generate embeddings.
        """
        try:
            # Create a pgAI vectorize job
            query = text(f"""
                SELECT pgai.vectorize_job(
                    '{table}',
                    '{id_column}',
                    {id_value},
                    '{text_column}',
                    '{embedding_column}',
                    'ollama',
                    '{self.model}'
                );
            """)
            
            result = db.execute(query).scalar()
            db.commit()
            return result
        except Exception as e:
            print(f"Error scheduling embedding generation: {str(e)}")
            raise e

embedding_service = EmbeddingService() 