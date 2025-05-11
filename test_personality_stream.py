import asyncio
import os
from dotenv import load_dotenv
import httpx
import json

async def test_streaming():
    # Set up the API URL (assuming the server is running locally)
    base_url = "http://localhost:8000"
    
    # Test user credentials
    username = "Shambhavi"  # Replace with a valid username from your database
    
    # Question to ask
    question = "What do you think about artificial intelligence?"
    
    # Create the request
    url = f"{base_url}/personalities/users/{username}/ask"
    payload = {
        "question": question,
        "stream": True,
        "multi_message": True
    }
    
    print(f"Sending streaming request to {url}")
    
    # Make the API call
    async with httpx.AsyncClient() as client:
        try:
            print(f"Sending request to {url} with payload: {payload}")
            response = await client.post(url, json=payload, timeout=60.0)
            
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response: {response.text}")
                return
            
            print("Stream started, receiving chunks:")
            
            # Process the streaming response
            async for line in response.aiter_lines():
                if line.strip():
                    print(f"Received chunk: {line[:50]}..." if len(line) > 50 else f"Received chunk: {line}")
                    try:
                        data = json.loads(line)
                        if data.get("done", False):
                            print("Stream completed successfully!")
                            break
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON: {line}")
            
            print("Test completed!")
            
        except Exception as e:
            print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the test
    asyncio.run(test_streaming()) 