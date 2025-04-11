from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

from app.api.routes import conversation
from app.db.database import Base, engine

# Load environment variables
load_dotenv()

# Drop and recreate database tables
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="AI Social App Backend",
    description="API for a personality-based social application",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(conversation.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Social App API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"} 