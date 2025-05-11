import asyncio
import os
from dotenv import load_dotenv
import json
from app.services.model_provider import ModelProvider

async def test_streaming():
    # Test messages to simulate chat
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    # Create a new model provider instance to ensure it picks up the correct env var
    model_provider = ModelProvider()
    
    print("Starting streaming test...")
    try:
        # Get the streaming object by awaiting the coroutine
        print("Calling model_provider.generate_chat...")
        streaming_obj = await model_provider.generate_chat(messages, stream=True)
        print(f"Type of returned object: {type(streaming_obj)}")
        
        # Check if it has the required methods
        has_aenter = hasattr(streaming_obj, '__aenter__')
        has_aexit = hasattr(streaming_obj, '__aexit__')
        has_aiter = hasattr(streaming_obj, '__aiter__')
        has_anext = hasattr(streaming_obj, '__anext__')
        
        print(f"Object implements __aenter__: {has_aenter}")
        print(f"Object implements __aexit__: {has_aexit}")
        print(f"Object implements __aiter__: {has_aiter}")
        print(f"Object implements __anext__: {has_anext}")
        
        if not (has_aenter and has_aexit):
            print("ERROR: Object doesn't implement the async context manager protocol")
            return
        
        # Use async with to test the context manager protocol
        print("Entering async with...")
        async with streaming_obj as response:
            print("Context manager entered successfully")
            # Iterate through the streaming response
            print("Starting iteration...")
            async for line in response:
                print(f"Received chunk: {line[:50]}...")
        
        print("Context manager exited successfully")
        print("Stream test completed successfully!")
    except Exception as e:
        print(f"Error during streaming test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Force CHAT_PROVIDER to gemini to test the Gemini provider
    os.environ["CHAT_PROVIDER"] = "gemini"
    print(f"Set CHAT_PROVIDER to: {os.environ.get('CHAT_PROVIDER')}")
    
    # Run the test
    asyncio.run(test_streaming()) 