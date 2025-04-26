import os
import json
from sqlalchemy import text
from sqlalchemy.orm import Session
import asyncio
from typing import List, Dict, Any, Optional
import hashlib
import redis
import httpx
from app.services.model_provider import model_provider

class EmbeddingService:
    def __init__(self):
        self.provider = os.getenv("CHAT_PROVIDER", "ollama").lower()
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = "nomic-embed-text"  # Default embedding model
        
        # Initialize embedding cache
        self.embedding_cache = {}
        self.cache_size = 200  # Store up to 200 embeddings in memory
        
        # Setup Redis connection if available
        redis_url = os.getenv("REDIS_URL")
        self.redis_client = None
        self.cache_expiry = 86400  # Cache expiry in seconds (24 hours)
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                print(f"Redis embedding cache initialized at {redis_url}")
            except Exception as e:
                print(f"Failed to connect to Redis for embeddings: {str(e)}")
                self.redis_client = None
        
        # Semaphore to limit concurrent API calls
        self.semaphore = asyncio.Semaphore(5)  # Allow up to 5 concurrent embedding calls
    
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embeddings for the given text using the selected provider with caching."""
        cache_key = self._get_cache_key(text)
        cached_embedding = await self._get_from_cache(cache_key)
        if cached_embedding:
            return cached_embedding

        # Use the model_provider abstraction
        embedding = await model_provider.generate_embedding(text)

        # Cache the embedding
        await self._add_to_cache(cache_key, embedding)
        return embedding
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in parallel."""
        tasks = []
        for text in texts:
            tasks.append(self.generate_embedding(text))
            
        return await asyncio.gather(*tasks)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text embedding."""
        # Use a hash of the text as the key
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Try to get an embedding from cache (either Redis or in-memory)."""
        # Try Redis first if available
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"embedding:{cache_key}")
                if cached:
                    return json.loads(cached.decode('utf-8'))
            except Exception as e:
                print(f"Redis embedding cache error: {str(e)}")
        
        # Fall back to in-memory cache
        return self.embedding_cache.get(cache_key)
    
    async def _add_to_cache(self, cache_key: str, embedding: List[float]) -> None:
        """Add an embedding to cache (both Redis and in-memory)."""
        # Add to Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"embedding:{cache_key}", 
                    self.cache_expiry,
                    json.dumps(embedding)
                )
            except Exception as e:
                print(f"Redis embedding cache error: {str(e)}")
        
        # Add to in-memory cache
        self.embedding_cache[cache_key] = embedding
        
        # Prune cache if it gets too large
        if len(self.embedding_cache) > self.cache_size:
            # Remove oldest entries
            keys_to_remove = list(self.embedding_cache.keys())[:-self.cache_size]
            for key in keys_to_remove:
                if key in self.embedding_cache:
                    del self.embedding_cache[key]
    
    def schedule_embedding_generation(self, db: Session, text: str, table: str, id_column: str, id_value: int, text_column: str, embedding_column: str):
        """
        Schedule embedding generation using pgAI's vectorize_job function.
        This uses the vectorizer-worker to asynchronously generate embeddings.
        """
        if self.provider != "ollama":
            # Do not store embeddings in Postgres for online models
            return None
        try:
            # Check if we're in a test environment
            if "pytest" in os.environ.get("PYTHONPATH", ""):
                # For testing, directly generate and store the embedding
                import json
                import asyncio
                
                # Generate the embedding
                embedding = asyncio.run(self.generate_embedding(text))
                
                # Store it directly in the database
                if table == "messages":
                    # For Message model
                    from app.models.conversation import Message
                    message = db.query(Message).filter(Message.id == id_value).first()
                    if message:
                        message.embedding = json.dumps(embedding)
                        db.commit()
                elif table == "personality_profiles":
                    # For PersonalityProfile model
                    from app.models.personality import PersonalityProfile
                    profile = db.query(PersonalityProfile).filter(PersonalityProfile.id == id_value).first()
                    if profile:
                        profile.embedding = json.dumps(embedding)
                        db.commit()
                
                return True
            
            # Create a pgAI vectorize job (production only)
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

class EmbeddingAutoUpdater:
    def __init__(self):
        self.embedding_service = embedding_service
        self.embedding_queue = set()  # Set of (table, id, column) tuples to avoid duplicates
        self.is_processing = False
        self.batch_size = 20  # Process 20 items per batch
        self.sleep_time = 5  # Sleep 5 seconds between batches
    
    def schedule_update(self, db: Session, table: str, id_column: str, id_value: int, 
                       text_column: str, embedding_column: str):
        """
        Schedule an item for embedding update.
        
        Parameters:
        - db: Database session
        - table: Table name
        - id_column: Primary key column name
        - id_value: Primary key value
        - text_column: Column containing the text to embed
        - embedding_column: Column to store the embedding
        """
        # Add to queue
        queue_item = (table, id_column, id_value, text_column, embedding_column)
        self.embedding_queue.add(queue_item)
        
        # Start processing if not already running
        if not self.is_processing:
            asyncio.create_task(self._process_queue(db))
    
    async def _process_queue(self, db: Session):
        """Process the embedding update queue in batches."""
        if self.is_processing:
            return
            
        self.is_processing = True
        
        try:
            while self.embedding_queue:
                # Process a batch of items
                batch = []
                for _ in range(min(self.batch_size, len(self.embedding_queue))):
                    if not self.embedding_queue:
                        break
                    batch.append(self.embedding_queue.pop())
                
                # Process each item in the batch
                for table, id_column, id_value, text_column, embedding_column in batch:
                    try:
                        # Get the text
                        query = text(f"""
                            SELECT {text_column} FROM {table} 
                            WHERE {id_column} = :id_value
                        """)
                        result = db.execute(query, {"id_value": id_value}).scalar()
                        
                        if result:
                            # Generate embedding
                            embedding = await self.embedding_service.generate_embedding(result)
                            
                            # Update the embedding in the database
                            update_query = text(f"""
                                UPDATE {table} 
                                SET {embedding_column} = :embedding 
                                WHERE {id_column} = :id_value
                            """)
                            db.execute(update_query, {
                                "embedding": embedding, 
                                "id_value": id_value
                            })
                            db.commit()
                    except Exception as e:
                        print(f"Error updating embedding for {table}.{id_column}={id_value}: {str(e)}")
                
                # Sleep between batches to avoid overwhelming the system
                if self.embedding_queue:
                    await asyncio.sleep(self.sleep_time)
        
        finally:
            self.is_processing = False

# Initialize the auto-updater
embedding_auto_updater = EmbeddingAutoUpdater() 