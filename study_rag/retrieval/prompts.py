def construct_rag_prompt(
    subject: str,
    query: str,
    context_text: str,
    chat_history: str = ""
) -> str:
    """Construct the RAG prompt with the given subject, user query, context text, and optional conversation history."""
    
    history_section = ""
    if chat_history:
        history_section = f"""---
Conversation History (for context only):
{chat_history}
---

"""
    
    return f"""You are an expert, friendly college-level teaching assistant for the subject '{subject}'. 
Your task is to answer the user's question accurately using the provided context chunks from their study materials.

Requirements:
1. Always base your response primarily on the provided context. 
2. Explicitly cite your sources using the format: (Filename, Page X). For example, "According to [Unit 3- CPU Scheduling Algorithms (1).pdf](file:///path/to/Unit 3...), page 12..."
3. If the context does not contain the information, state: "I couldn't find a direct answer in the provided textbook notes, but here is general knowledge on the topic:" and then explain it using your general knowledge.
4. Keep explanations clear, academic, yet easy to understand. Use bullet points, code blocks, or markdown tables where helpful.
5. If the conversation history is provided, ensure your response is consistent with previous answers and addresses any follow-up nuances.

{history_section}Context:
{context_text}

Question:
{query}
"""
