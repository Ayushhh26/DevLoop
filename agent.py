"""
Multi-Agent System: Coder + Critic Loop

This module implements the agentic loop with:
- Coder Agent: Generates code fixes based on context and queries
- Critic Agent: Reviews code against quality rules
- Main loop: Orchestrates interaction with rejection feedback
"""

import os
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Rate limiting for Groq Free Tier: 30 requests per minute = ~2 seconds per request
# We'll use a simple time-based rate limiter
_last_request_time = None
_min_request_interval = 2.1  # Slightly more than 2 seconds to be safe (30 req/min = 2 sec/req)


def get_llm(provider: Optional[str] = None) -> BaseChatModel:
    """
    Factory function to get LLM instance based on provider.
    
    Args:
        provider: LLM provider name ("GROQ" or "DEEPSEEK"). If None, reads from env.
    
    Returns:
        Chat model instance
    
    Raises:
        ValueError: If provider is not supported
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "GROQ").upper()
    
    if provider == "GROQ":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=api_key,
        )
    
    elif provider == "DEEPSEEK":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        return ChatOpenAI(
            base_url="https://api.deepseek.com",
            model="deepseek-coder",
            api_key=api_key,
            temperature=0.7,
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: GROQ, DEEPSEEK")


def get_default_rules() -> List[str]:
    """
    Get default code review rules.
    These rules are GENERIC and work for any repository.
    """
    return [
        # Style rules
        "All functions must have type hints",
        "No bare 'except:' clauses - must specify exception types",
        # Context-aware rules (generic, not repo-specific)
        "Only use APIs, methods, and imports that appear in the retrieved code context",
        "Do not invent or hallucinate methods, properties, or classes not shown in the codebase",
        "If fixing an existing function, match the original function signature",
        "Follow the coding patterns and style shown in the retrieved context",
    ]


def rate_limit_delay(provider: Optional[str] = None):
    """
    Simple rate limiter to prevent exceeding API limits.
    Groq Free Tier: 30 requests/minute = ~2 seconds between requests.
    """
    global _last_request_time
    
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "GROQ").upper()
    
    # Only rate limit for Groq (free tier has strict limits)
    if provider == "GROQ":
        current_time = time.time()
        
        if _last_request_time is not None:
            time_since_last_request = current_time - _last_request_time
            
            if time_since_last_request < _min_request_interval:
                sleep_time = _min_request_interval - time_since_last_request
                time.sleep(sleep_time)
        
        _last_request_time = time.time()


def call_llm_with_retry(llm: BaseChatModel, messages: List, provider: Optional[str] = None, max_retries: int = 3) -> str:
    """
    Call LLM with rate limiting and retry logic for rate limit errors.
    
    Args:
        llm: Language model instance
        messages: List of messages to send
        provider: LLM provider name
        max_retries: Maximum number of retries
    
    Returns:
        Response content
    
    Raises:
        Exception: If all retries fail
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "GROQ").upper()
    
    for attempt in range(max_retries):
        try:
            # Apply rate limiting
            rate_limit_delay(provider)
            
            # Make the API call
            response = llm.invoke(messages)
            return response.content.strip()
        
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error
            if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
                if attempt < max_retries - 1:
                    # Exponential backoff: wait longer on each retry
                    wait_time = (2 ** attempt) * _min_request_interval
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Rate limit exceeded after {max_retries} retries. Please wait a few minutes before trying again.")
            else:
                # For non-rate-limit errors, raise immediately
                raise
    
    raise Exception("Failed to get response from LLM after retries")


def coder_agent(llm: BaseChatModel, query: str, context: List[str], provider: Optional[str] = None) -> str:
    """
    Coder Agent: Generates code fixes based on user query and retrieved context.
    
    Args:
        llm: Language model instance
        query: User's issue/question
        context: Retrieved code chunks (list of strings)
        provider: LLM provider name (for rate limiting)
    
    Returns:
        Generated code fix
    """
    context_text = "\n\n---\n\n".join(context)
    
    system_prompt = """You are an expert developer. Given the retrieved code context and a user issue, write a fix.

CRITICAL RULES:
1. ONLY use APIs, methods, imports, and patterns that appear in the retrieved context
2. Do NOT invent or hallucinate methods/properties that don't exist in the codebase
3. If fixing an existing function, MATCH the original function signature exactly
4. Follow the coding style and patterns shown in the retrieved code
5. If unsure about an API, use what's shown in the context, not general knowledge

Output ONLY the corrected code block with no explanations. Do not include markdown code fences (```python or ```).
Focus on fixing the specific issue mentioned by the user."""
    
    user_prompt = f"""Retrieved code context from the repository:
{context_text}

User Issue:
{query}

IMPORTANT: Base your fix ONLY on the APIs and patterns shown in the context above.
Provide the fixed code:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    
    return call_llm_with_retry(llm, messages, provider)


def critic_agent(llm: BaseChatModel, draft_code: str, rules: List[str], provider: Optional[str] = None) -> Dict[str, str]:
    """
    Critic Agent: Reviews code against quality rules.
    
    Args:
        llm: Language model instance
        draft_code: Code to review
        rules: List of review rules
        provider: LLM provider name (for rate limiting)
    
    Returns:
        Dictionary with 'status' ('APPROVE' or 'REJECT') and 'reason'
    """
    rules_text = "\n".join(f"- {rule}" for rule in rules)
    
    system_prompt = """You are a Senior Tech Lead reviewing code. Review the proposed fix against the given rules.
If it violates ANY rule, output 'REJECT: <specific reason>'. 
If it passes all rules, output 'APPROVE'.
Be strict but fair."""
    
    user_prompt = f"""Review Rules:
{rules_text}

Proposed Code:
{draft_code}

Your review (APPROVE or REJECT: <reason>):"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    
    critique = call_llm_with_retry(llm, messages, provider)
    
    # Parse response
    if critique.upper().startswith("APPROVE"):
        return {"status": "APPROVE", "reason": critique}
    elif critique.upper().startswith("REJECT"):
        reason = critique.split(":", 1)[1].strip() if ":" in critique else critique
        return {"status": "REJECT", "reason": reason}
    else:
        # Default to reject if unclear
        return {"status": "REJECT", "reason": f"Unclear review: {critique}"}


def run_agent_loop(query: str, context: List[str], rules: Optional[List[str]] = None, provider: Optional[str] = None, max_iterations: int = 3) -> Dict:
    """
    Main agentic loop: Coder generates fix, Critic reviews, loops on rejection.
    
    Args:
        query: User's issue/question
        context: Retrieved code chunks
        rules: Code review rules (defaults to get_default_rules())
        provider: LLM provider ("GROQ" or "DEEPSEEK")
        max_iterations: Maximum number of iterations (default: 3)
    
    Returns:
        Dictionary with:
        - draft_code: Final code
        - critique: Critic's review
        - final_status: "APPROVE" or "REJECT"
        - iterations: Number of iterations
    """
    if rules is None:
        rules = get_default_rules()
    
    # Get LLM instance
    llm = get_llm(provider)
    
    draft_code = ""
    critique = ""
    final_status = "REJECT"
    iterations = 0
    
    for iteration in range(max_iterations):
        iterations = iteration + 1
        
        # Coder generates fix
        if iteration == 1:
            # First iteration: generate from original query
            draft_code = coder_agent(llm, query, context, provider)
        else:
            # Subsequent iterations: incorporate critique feedback
            feedback_query = f"{query}\n\nPrevious attempt was rejected. Reason: {critique}\n\nPlease fix the issues and provide corrected code."
            draft_code = coder_agent(llm, feedback_query, context, provider)
        
        # Critic reviews
        review = critic_agent(llm, draft_code, rules, provider)
        critique = review["reason"]
        final_status = review["status"]
        
        # If approved, break loop
        if final_status == "APPROVE":
            break
    
    return {
        "draft_code": draft_code,
        "critique": critique,
        "final_status": final_status,
        "iterations": iterations,
    }


if __name__ == "__main__":
    # Example usage
    test_query = "Fix the login function to handle errors properly"
    test_context = [
        "def login(username, password):\n    try:\n        auth_user(username, password)\n    except:\n        return None",
    ]
    
    result = run_agent_loop(test_query, test_context)
    print(f"Status: {result['final_status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Code: {result['draft_code']}")
