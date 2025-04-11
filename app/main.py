from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.api.routes import conversation
from app.db.database import Base, engine, get_db

# Load environment variables
load_dotenv()

# Create database tables if they don't exist
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

@app.post("/admin/reset-database")
def reset_database(db: Session = Depends(get_db)):
    """Admin endpoint to drop and recreate all database tables. USE WITH CAUTION!"""
    # Close all connections
    db.close()
    
    # Drop and recreate tables
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    return {"message": "Database has been reset. All data has been deleted."} 