import time
import random
from rag_tutor.llm.providers import get_embedding_provider


def get_embeddings_batch(texts: list[str], api_key: str = None, provider_name: str = None) -> list[list[float]]:
    """Get embeddings for a list of texts using the configured provider in batches with backoff retry."""
    provider = get_embedding_provider(name=provider_name, api_key=api_key)
    batch_size = 64
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        delay = 2.0
        max_retries = 6
        result = None

        for attempt in range(max_retries):
            try:
                result = provider.embed(batch)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
                    sleep_time = delay + random.uniform(0.1, 0.5)
                    print(f"Rate limited on batch. Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    delay *= 2
                else:
                    raise e
        else:
            raise RuntimeError(f"Failed to generate embeddings for batch after {max_retries} retries due to rate limiting.")

        if result:
            embeddings.extend(result)
        time.sleep(1.2)

    return embeddings


def get_embedding(text: str, api_key: str = None, provider_name: str = None) -> list[float]:
    """Get embedding for a single text string."""
    results = get_embeddings_batch([text], api_key=api_key, provider_name=provider_name)
    return results[0]