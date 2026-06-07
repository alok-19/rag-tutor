import time
from typing import Generator, Callable, Tuple
from dataclasses import dataclass
from rag_tutor.llm.providers import get_provider
from rag_tutor.config import MODEL_CONFIG, LLM_PROVIDER


@dataclass
class ChatResponse:
    content: str
    fallback_used: bool = False


def _is_transient(e: Exception) -> bool:
    msg = str(e)
    return "503" in msg or "UNAVAILABLE" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg.lower()


def generate_response_stream(
    prompt: str,
    api_key: str = None,
    status_callback: Callable[[str], None] = None,
    provider_name: str = None,
) -> Generator[Tuple[str, bool], None, None]:
    """Generate content stream with retries and fallback across providers.

    Yields tuples of (text_chunk, fallback_used).
    """
    provider = get_provider(name=provider_name, api_key=api_key)
    cfg = MODEL_CONFIG.get(provider.name, MODEL_CONFIG["gemini"])
    model_name = cfg["primary"]
    fallback_model = cfg.get("fallback")
    fallback_used = False

    # Attempt 1: Primary Model
    try:
        for chunk in provider.generate_stream(prompt, model=model_name):
            yield chunk, fallback_used
        return
    except Exception as e:
        if not _is_transient(e):
            raise e
        if status_callback:
            status_callback("⏳ Our AI assistant is currently experiencing high demand. Retrying your request automatically in 3 seconds...")
        time.sleep(3.0)
        if status_callback:
            status_callback("")

    # Attempt 2: Primary Model Retry
    try:
        for chunk in provider.generate_stream(prompt, model=model_name):
            yield chunk, fallback_used
        return
    except Exception as e2:
        if not _is_transient(e2):
            raise e2

        # Attempt 3: Fallback Model (same provider)
        if fallback_model:
            fallback_used = True
            if status_callback:
                status_callback(f"⏳ Still experiencing high demand. Falling back to secondary model ({fallback_model})...")
            time.sleep(1.5)
            if status_callback:
                status_callback("")

            for chunk in provider.generate_stream(prompt, model=fallback_model):
                yield chunk, fallback_used
            return
        else:
            raise e2