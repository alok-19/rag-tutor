from rag_tutor.llm.embeddings import get_embedding
from rag_tutor.retrieval.query_expansion import expand_query
from rag_tutor.retrieval.vector_store import query_vector_store, RetrievedSource
from rag_tutor.config import DB_PATH, COLLECTION_NAME
from pathlib import Path

def build_chat_history(messages: list[dict], max_turns: int = 3) -> str:
    """Convert the last N message turns into a formatted chat history string.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        max_turns: Maximum number of conversation turns (user + assistant pairs) to include.
        
    Returns:
        Formatted chat history string, or empty string if no messages.
    """
    if not messages:
        return ""
    
    # Take the last max_turns * 2 messages (pairs of user + assistant)
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = msg["role"].capitalize()
        content = msg["content"]
        lines.append(f"{role}: {content}")
    
    return "\n".join(lines)

def disambiguate_query(query: str, messages: list[dict]) -> str:
    """Transform a potentially ambiguous follow-up query into a standalone query
    using context from the previous assistant response.
    
    This is useful when users ask follow-ups like 'Explain more' or 'Why is that?'
    without referencing the original topic.
    """
    query_lower = query.lower().strip()
    
    # Very short queries (< 15 chars) are likely follow-ups
    is_very_short = len(query_lower) < 15
    
    # Explicit follow-up prefixes
    follow_up_prefixes = [
        "explain more", "tell me more", "elaborate", "clarify",
        "what about", "expand on", "could you", "can you"
    ]
    is_explicit_follow_up = any(query_lower.startswith(prefix) for prefix in follow_up_prefixes)
    
    # Queries containing pronouns that refer to prior context
    pronouns = [" it ", " that ", " this ", " them ", " those ", " they "]
    has_pronoun = any(pronoun in f" {query_lower} " for pronoun in pronouns)
    
    # Queries starting with vague starters
    vague_starters = ["why ", "how ", "and ", "but ", "so "]
    is_vague = any(query_lower.startswith(starter) for starter in vague_starters)
    
    should_disambiguate = is_very_short or is_explicit_follow_up or (has_pronoun and is_vague)
    
    if should_disambiguate and messages:
        # Find the last assistant message to extract context
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                # Use the first sentence (up to 200 chars) as disambiguation context
                content = msg["content"]
                first_sentence = content.split(".")[0].strip()[:200]
                if first_sentence:
                    return f"Regarding: {first_sentence}. {query}"
                break
    
    return query

def retrieve_context_with_memory(
    query: str,
    subject: str,
    messages: list[dict],
    api_key: str = None,
    n_results: int = 4,
    db_path: Path = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    use_reranker: bool = None
) -> tuple[list[RetrievedSource], str]:
    """Perform retrieval with conversational memory support.
    
    Returns:
        Tuple of (retrieved_sources, disambiguated_query)
    """
    disambiguated = disambiguate_query(query, messages)

    from rag_tutor.config import ENABLE_RERANKER, RERANKER_INITIAL_K
    from rag_tutor.retrieval.rag_service import _maybe_rerank

    expanded = expand_query(disambiguated)
    initial_k = max(n_results, RERANKER_INITIAL_K) if (use_reranker if use_reranker is not None else ENABLE_RERANKER) else n_results
    initial_k = min(initial_k, 50)

    query_embedding = get_embedding(expanded, api_key=api_key)
    candidates = query_vector_store(
        query_embedding=query_embedding,
        subject=subject,
        n_results=initial_k,
        db_path=db_path,
        collection_name=collection_name
    )
    sources = _maybe_rerank(expanded, candidates, use_reranker, top_k=n_results)
    return sources, disambiguated
