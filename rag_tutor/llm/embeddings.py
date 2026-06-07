import time
import random
from google.genai.errors import APIError
from google.genai import types
from rag_tutor.config import EMBEDDING_MODEL
from rag_tutor.llm.gemini_client import get_genai_client

def get_embeddings_batch(texts: list[str], api_key: str = None) -> list[list[float]]:
    """Get embeddings for a list of texts using Gemini API in batches with backoff retry."""
    client = get_genai_client(api_key=api_key)
    embeddings = []
    # Larger batch size to minimize requests count
    batch_size = 64
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Explicitly wrap strings in types.Content to ensure batch items are treated as separate inputs
        contents = [types.Content(parts=[types.Part.from_text(text=txt)]) for txt in batch]
        
        # Exponential backoff retry loop
        delay = 2.0
        max_retries = 6
        response = None
        
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=contents
                )
                break
            except APIError as e:
                if e.code == 429:
                    sleep_time = delay + random.uniform(0.1, 0.5)
                    print(f"Rate limited (429) on batch. Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    delay *= 2
                else:
                    raise e
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    sleep_time = delay + random.uniform(0.1, 0.5)
                    print(f"Rate limited (RESOURCE_EXHAUSTED) on batch. Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    delay *= 2
                else:
                    raise e
        else:
            raise RuntimeError(f"Failed to generate embeddings for batch after {max_retries} retries due to rate limiting.")
            
        if response:
            embeddings.extend([e.values for e in response.embeddings])
            
        # Be polite to the rate limit by adding a small spacing delay
        time.sleep(1.2)
            
    return embeddings

def get_embedding(text: str, api_key: str = None) -> list[float]:
    """Get embedding for a single text string."""
    results = get_embeddings_batch([text], api_key=api_key)
    return results[0]
