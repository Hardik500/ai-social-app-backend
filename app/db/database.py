from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Enable pgAI extension on first connect
@event.listens_for(engine, "first_connect")
def enable_extensions(connect, connection_record):
    connect.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connect.execute("CREATE EXTENSION IF NOT EXISTS pgai")  # For TimescaleDB pgAI 