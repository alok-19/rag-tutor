import time
from typing import Generator, Callable, Tuple
from dataclasses import dataclass
from study_rag.config import PRIMARY_GENERATION_MODEL, FALLBACK_GENERATION_MODEL
from study_rag.llm.gemini_client import get_genai_client

@dataclass
class ChatResponse:
    content: str
    fallback_used: bool = False

def generate_response_stream(
    prompt: str,
    api_key: str = None,
    status_callback: Callable[[str], None] = None
) -> Generator[Tuple[str, bool], None, None]:
    """Generate content stream with retries and fallback.
    Yields tuples of (text_chunk, fallback_used).
    """
    client = get_genai_client(api_key=api_key)
    model_name = PRIMARY_GENERATION_MODEL
    fallback_used = False
    
    def is_transient(e: Exception) -> bool:
        msg = str(e)
        return "503" in msg or "UNAVAILABLE" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg
        
    # Attempt 1: Primary Model
    try:
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=prompt
        )
        for chunk in response_stream:
            yield chunk.text, fallback_used
        return
    except Exception as e:
        if not is_transient(e):
            raise e
            
        if status_callback:
            status_callback("⏳ Our AI assistant is currently experiencing high demand. Retrying your request automatically in 3 seconds...")
        time.sleep(3.0)
        if status_callback:
            status_callback("")  # Clear status
            
    # Attempt 2: Primary Model Retry
    try:
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=prompt
        )
        for chunk in response_stream:
            yield chunk.text, fallback_used
        return
    except Exception as e2:
        if not is_transient(e2):
            raise e2
            
        # Attempt 3: Fallback Model
        fallback_used = True
        model_name = FALLBACK_GENERATION_MODEL
        if status_callback:
            status_callback(f"⏳ Still experiencing high demand. Falling back to secondary model ({model_name})...")
        time.sleep(1.5)
        if status_callback:
            status_callback("")  # Clear status
            
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=prompt
        )
        for chunk in response_stream:
            yield chunk.text, fallback_used
