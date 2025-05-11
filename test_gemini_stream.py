import asyncio
import os
from dotenv import load_dotenv
import json
from app.services.model_providers.gemini_provider import GeminiProvider
from app.services.model_providers.gemini_provider import GeminiStreamingResponse

async def test_gemini_streaming():
    # Load environment variables
    load_dotenv()
    
    # Check if GOOGLE_API_KEY is set
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable must be set")
        return
    
    # Test messages to simulate chat
    contents = [
        "You are a helpful assistant.",
        "Tell me a short story about a space adventure"
    ]
    
    # Create a Gemini provider instance
    provider = GeminiProvider()
    
    print("Starting Gemini streaming test...")
    try:
        # Get the streaming response object
        print("Creating streaming response object...")
        streaming_obj = GeminiStreamingResponse(
            contents=contents,
            client=provider.client,
            model_name=provider.chat_model,
            format_json=False
        )
        
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
            full_response = ""
            async for line in response:
                chunk = json.loads(line)
                content = chunk["message"]["content"]
                full_response += content
                print(f"Received chunk: {content[:50]}..." if len(content) > 50 else f"Received chunk: {content}")
        
        print("\nFull response:")
        print(full_response)
        print("\nContext manager exited successfully")
        print("Stream test completed successfully!")
    except Exception as e:
        print(f"Error during streaming test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_gemini_streaming()) 