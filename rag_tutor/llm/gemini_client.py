import os
from google import genai

_genai_client = None
_last_api_key = None

def get_genai_client(api_key: str = None) -> genai.Client:
    """Get or initialize the Gemini API client dynamically.
    Re-initializes if a different API key is provided.
    """
    global _genai_client, _last_api_key
    
    current_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not current_key:
        raise ValueError(
            "GEMINI_API_KEY/GOOGLE_API_KEY environment variable not set.\n"
            "Please copy .env.template to .env and fill in your key, or set it via the sidebar UI."
        )
        
    if _genai_client is None or current_key != _last_api_key:
        try:
            _genai_client = genai.Client(api_key=current_key)
            _last_api_key = current_key
        except Exception as e:
            raise RuntimeError(f"Error initializing Gemini client: {e}")
            
    return _genai_client
