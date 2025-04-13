#!/usr/bin/env python3
"""
Script to run SQL migrations for the Social AI App Backend.
This applies SQL migration files in the migrations directory.
"""

import os
import glob
import argparse
import sqlalchemy
from sqlalchemy import text
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_migration(engine, migration_file):
    """Run a single SQL migration file"""
    print(f"Running migration: {os.path.basename(migration_file)}")
    
    try:
        with open(migration_file, 'r') as f:
            sql = f.read()
            
        with engine.begin() as conn:
            conn.execute(text(sql))
            
        print(f"✅ Successfully applied migration: {os.path.basename(migration_file)}")
        return True
        
    except Exception as e:
        print(f"❌ Error applying migration {os.path.basename(migration_file)}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument("--file", help="Run a specific migration file")
    args = parser.parse_args()
    
    # Get database URL from environment variable
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        return
        
    # Create engine
    engine = sqlalchemy.create_engine(database_url)
    
    # Create migrations table if it doesn't exist
    create_migration_table_sql = """
    CREATE TABLE IF NOT EXISTS migrations (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255) NOT NULL UNIQUE,
        applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """
    
    with engine.begin() as conn:
        conn.execute(text(create_migration_table_sql))
    
    # Get all migration files
    migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")
    
    if args.file:
        # Run a specific migration file
        migration_file = os.path.join(migrations_dir, args.file)
        if not os.path.exists(migration_file):
            print(f"❌ Migration file not found: {args.file}")
            return
            
        migration_files = [migration_file]
    else:
        # Get all .sql files in the migrations directory
        migration_files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
    
    # Get already applied migrations
    with engine.connect() as conn:
        result = conn.execute(text("SELECT filename FROM migrations"))
        applied_migrations = {row[0] for row in result}
    
    # Apply migrations that haven't been applied yet
    for migration_file in migration_files:
        filename = os.path.basename(migration_file)
        
        if filename in applied_migrations:
            print(f"⏭️ Skipping already applied migration: {filename}")
            continue
            
        success = run_migration(engine, migration_file)
        
        if success:
            # Record the migration as applied
            with engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO migrations (filename) VALUES (:filename)"),
                    {"filename": filename}
                )

if __name__ == "__main__":
    main() 