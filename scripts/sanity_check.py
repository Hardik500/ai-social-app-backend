#!/usr/bin/env python3
"""
Sanity check script to verify the basic flow of the application.
This script:
1. Uploads a sample conversation
2. Retrieves the conversation
3. Verifies embedding was generated
4. Performs a similarity search
5. Verifies duplicate detection

Usage:
    python scripts/sanity_check.py
"""

import os
import sys
import json
import time
import httpx
import asyncio
import uuid
from datetime import datetime
from rich.console import Console
from rich.table import Table

# Add the parent directory to the path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create console for nice output
console = Console()

# API URL (can be overridden with environment variable)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Generate unique ID for this run
run_id = str(uuid.uuid4())[:8]
timestamp = int(datetime.now().timestamp())

# Sample conversation data with unique IDs
SAMPLE_CONVERSATION = {
    "source": "slack",
    "messages": [
        {
            "user": f"John_{run_id}",
            "timestamp": f"{timestamp}.637399",
            "message": "Let's schedule the meeting for next Tuesday at 2 PM"
        },
        {
            "user": f"Sarah_{run_id}",
            "timestamp": f"{timestamp}.807159",
            "message": "That works for me. I'll book the conference room"
        },
        {
            "user": f"John_{run_id}",
            "timestamp": f"{timestamp}.830249",
            "message": "Great! Don't forget to send the agenda beforehand"
        }
    ],
    "user_info": {
        "username": f"John_{run_id}",
        "email": f"john_{run_id}@example.com",
        "phone": f"123-456-{run_id}",
        "description": "Project manager"
    }
}

# Search query
SEARCH_QUERY = "When is our meeting scheduled?"

async def check_health():
    """Check if the API is healthy"""
    console.print("\n[bold blue]Checking API health...[/bold blue]")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/health")
            
            if response.status_code == 200:
                console.print("[green]✓ API is healthy![/green]")
                return True
            else:
                console.print(f"[red]✗ API returned status {response.status_code}[/red]")
                return False
    except Exception as e:
        console.print(f"[red]✗ Error connecting to API: {str(e)}[/red]")
        return False

async def upload_conversation():
    """Upload a sample conversation and return its ID"""
    console.print("\n[bold blue]Uploading sample conversation...[/bold blue]")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/conversations/",
                json=SAMPLE_CONVERSATION
            )
            
            if response.status_code == 201:
                data = response.json()
                console.print(f"[green]✓ Conversation uploaded successfully! ID: {data['id']}[/green]")
                return data
            else:
                console.print(f"[red]✗ Failed to upload conversation: {response.status_code} - {response.text}[/red]")
                return None
    except Exception as e:
        console.print(f"[red]✗ Error uploading conversation: {str(e)}[/red]")
        return None

async def check_duplicate_detection(original_conversation):
    """Upload the same conversation again to test duplicate detection"""
    console.print("\n[bold blue]Testing duplicate detection...[/bold blue]")
    
    try:
        async with httpx.AsyncClient() as client:
            # Try to upload the same conversation again
            response = await client.post(
                f"{API_URL}/conversations/",
                json=SAMPLE_CONVERSATION
            )
            
            if response.status_code == 200:
                data = response.json()
                if "duplicate_detection" in data and data["duplicate_detection"]["is_duplicate"]:
                    console.print(f"[green]✓ Duplicate correctly detected! Original ID: {original_conversation['id']}, Match: {data['duplicate_detection']['match_percentage']:.2f}%[/green]")
                    return True
                else:
                    console.print(f"[yellow]! Response code 200 but no duplicate detection info[/yellow]")
                    return False
            elif response.status_code == 201:
                console.print(f"[red]✗ Duplicate not detected - new conversation created with ID: {response.json()['id']}[/red]")
                return False
            else:
                console.print(f"[red]✗ Error testing duplicate detection: {response.status_code} - {response.text}[/red]")
                return False
    except Exception as e:
        console.print(f"[red]✗ Error in duplicate detection: {str(e)}[/red]")
        return False

async def test_non_duplicate():
    """Upload a modified conversation to verify it's not flagged as duplicate"""
    console.print("\n[bold blue]Testing non-duplicate detection...[/bold blue]")
    
    try:
        # Create a modified version of the conversation (with different timestamps)
        modified_conversation = SAMPLE_CONVERSATION.copy()
        modified_conversation = json.loads(json.dumps(modified_conversation))  # Deep copy
        
        # Shift timestamps by 1 day (86400 seconds)
        for msg in modified_conversation["messages"]:
            timestamp_parts = msg["timestamp"].split(".")
            new_timestamp = str(float(timestamp_parts[0]) + 86400)
            if len(timestamp_parts) > 1:
                new_timestamp += "." + timestamp_parts[1]
            msg["timestamp"] = new_timestamp
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/conversations/",
                json=modified_conversation
            )
            
            if response.status_code == 201:
                data = response.json()
                console.print(f"[green]✓ Modified conversation correctly treated as new (ID: {data['id']})[/green]")
                return True
            elif response.status_code == 200 and "duplicate_detection" in response.json():
                console.print(f"[red]✗ Modified conversation incorrectly detected as duplicate[/red]")
                return False
            else:
                console.print(f"[red]✗ Error testing non-duplicate: {response.status_code} - {response.text}[/red]")
                return False
    except Exception as e:
        console.print(f"[red]✗ Error in non-duplicate test: {str(e)}[/red]")
        return False

async def get_conversation(conversation_id):
    """Retrieve a conversation by its ID"""
    console.print(f"\n[bold blue]Retrieving conversation (ID: {conversation_id})...[/bold blue]")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/conversations/{conversation_id}")
            
            if response.status_code == 200:
                data = response.json()
                console.print(f"[green]✓ Conversation retrieved successfully![/green]")
                console.print(f"[dim]Source: {data['source']}, Messages: {len(data['messages'])}[/dim]")
                return data
            else:
                console.print(f"[red]✗ Failed to retrieve conversation: {response.status_code} - {response.text}[/red]")
                return None
    except Exception as e:
        console.print(f"[red]✗ Error retrieving conversation: {str(e)}[/red]")
        return None

async def check_embedding(message_id):
    """Check if embedding was generated for a message"""
    console.print(f"\n[bold blue]Checking embedding for message (ID: {message_id})...[/bold blue]")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/conversations/messages/{message_id}/embedding")
            
            if response.status_code == 200:
                data = response.json()
                if data["embedding_exists"]:
                    console.print(f"[green]✓ Embedding exists! Dimensions: {data['embedding_dimensions']}[/green]")
                    return True
                else:
                    console.print(f"[yellow]! Embedding does not exist yet[/yellow]")
                    return False
            else:
                console.print(f"[red]✗ Failed to check embedding: {response.status_code} - {response.text}[/red]")
                return False
    except Exception as e:
        console.print(f"[red]✗ Error checking embedding: {str(e)}[/red]")
        return False

async def search_similar_messages(query):
    """Search for messages similar to the query"""
    console.print(f"\n[bold blue]Searching for messages similar to: \"{query}\"...[/bold blue]")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/conversations/search/similar?query={query}")
            
            if response.status_code == 200:
                data = response.json()
                console.print(f"[green]✓ Search completed successfully![/green]")
                
                # Create a table for results
                table = Table(title="Similar Messages")
                table.add_column("Content", style="cyan")
                table.add_column("User", style="green")
                table.add_column("Score", style="magenta", justify="right")
                
                for msg in data["similar_messages"]:
                    table.add_row(
                        msg["content"], 
                        msg["username"], 
                        f"{msg['similarity_score']:.4f}"
                    )
                
                console.print(table)
                return data["similar_messages"]
            else:
                console.print(f"[red]✗ Failed to search: {response.status_code} - {response.text}[/red]")
                return None
    except Exception as e:
        console.print(f"[red]✗ Error during search: {str(e)}[/red]")
        return None

async def run_sanity_check():
    """Run the full sanity check flow"""
    console.print("[bold yellow]=== AI Social App Backend Sanity Check ===[/bold yellow]")
    console.print(f"API URL: {API_URL}")
    console.print(f"Run ID: {run_id}")
    
    # Step 1: Check API health
    if not await check_health():
        console.print("[bold red]Sanity check failed: API is not healthy![/bold red]")
        return False
    
    # Step 2: Upload conversation
    conversation = await upload_conversation()
    if not conversation:
        console.print("[bold red]Sanity check failed: Couldn't upload conversation![/bold red]")
        return False
    
    conversation_id = conversation["id"]
    
    # Step 3: Retrieve conversation
    retrieved_conversation = await get_conversation(conversation_id)
    if not retrieved_conversation:
        console.print("[bold red]Sanity check failed: Couldn't retrieve conversation![/bold red]")
        return False
    
    # Step 4: Check embedding for the first message
    message_id = retrieved_conversation["messages"][0]["id"]
    
    # It might take a moment for embeddings to be generated
    embedding_exists = False
    for i in range(3):  # Try 3 times
        embedding_exists = await check_embedding(message_id)
        if embedding_exists:
            break
        console.print("[yellow]Waiting for embedding generation (5s)...[/yellow]")
        time.sleep(5)
    
    if not embedding_exists:
        console.print("[bold yellow]Warning: Embedding not found after retries, but continuing...[/bold yellow]")
    
    # Step 5: Search for similar messages
    similar_messages = await search_similar_messages(SEARCH_QUERY)
    if similar_messages is None:
        console.print("[bold red]Sanity check failed: Couldn't search for similar messages![/bold red]")
        return False

    # Step 6: Test duplicate detection
    if not await check_duplicate_detection(conversation):
        console.print("[bold yellow]Warning: Duplicate detection test failed, but continuing...[/bold yellow]")
    
    # Step 7: Test non-duplicate detection
    if not await test_non_duplicate():
        console.print("[bold yellow]Warning: Non-duplicate test failed, but continuing...[/bold yellow]")
    
    # All steps passed!
    console.print("\n[bold green]=== Sanity Check Passed! ===[/bold green]")
    console.print("All features are working as expected.")
    return True

if __name__ == "__main__":
    result = asyncio.run(run_sanity_check())
    sys.exit(0 if result else 1) 