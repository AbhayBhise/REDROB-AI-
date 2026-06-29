"""
LLM-Powered Reasoning Generation Module

This module generates personalized, persuasive 2-sentence justifications for why 
each candidate fits the Senior AI Engineer role. It uses an LLM (OpenAI, Anthropic, 
or other provider via litellm) to craft human-like reasoning that goes beyond 
templated strings.

Key Features:
- Multi-provider support (OpenAI, Anthropic, Gemini, Cohere) via litellm
- Fast in-memory caching to avoid duplicate API calls
- Graceful fallback to deterministic reasoning if API key is missing/fails
- Timeout protection (30s per candidate)
- Production-ready error handling and logging
- Optional batch mode for efficiency (generate reasoning for multiple candidates)

Usage:
    from src.llm_reasoning import generate_llm_reasoning, batch_generate_llm_reasoning
    
    reasoning = generate_llm_reasoning(candidate, jd_text)
    # or fallback is automatic
    
    all_reasoning = batch_generate_llm_reasoning(top_100_candidates, jd_text)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    logger.warning(
        "litellm not installed. LLM reasoning disabled. "
        "Install via: pip install litellm"
    )


# Global in-memory cache to avoid duplicate API calls
_reasoning_cache: Dict[str, str] = {}


def _get_cache_key(candidate_id: str, jd_hash: str) -> str:
    """Generate a cache key for a (candidate, JD) pair."""
    return f"{candidate_id}:{jd_hash}"


def _hash_jd(jd_text: str) -> str:
    """Generate a short hash of the JD for cache keying."""
    return hashlib.md5(jd_text.encode()).hexdigest()[:8]


def _construct_llm_prompt(candidate: Dict[str, Any], jd_text: str) -> str:
    """
    Construct a detailed prompt for the LLM to generate reasoning.
    
    Args:
        candidate: Candidate profile dictionary
        jd_text: Job description text
    
    Returns:
        Formatted prompt string for the LLM
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    # Extract key candidate information
    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    
    # Extract top skills
    skills = []
    for s in candidate.get("skills", []):
        if s.get("proficiency") in ["advanced", "intermediate"]:
            skills.append(s.get("name", ""))
    top_skills = ", ".join(skills[:5])
    
    # Extract career highlights
    recent_roles = []
    for job in candidate.get("career_history", [])[:2]:
        if job.get("title"):
            recent_roles.append(f"{job.get('title')} at {job.get('company', 'Unknown')}")
    
    prompt = f"""
You are an expert recruiter evaluating a candidate for a Senior AI Engineer role.

Job Description (Key Requirements):
{jd_text}

Candidate Profile:
- Name: {candidate.get('candidate_id', 'Unknown')}
- Current Role: {title}
- Years of Experience: {yoe}
- Headline: {headline}
- Summary: {summary}
- Top Skills: {top_skills}
- Recent Roles: {'; '.join(recent_roles) if recent_roles else 'N/A'}
- Open to Work: {signals.get('open_to_work_flag', False)}
- Notice Period: {signals.get('notice_period_days', 'N/A')} days
- Location: {profile.get('location', 'Unknown')}

Your Task:
Generate a SHORT, PERSUASIVE 2-sentence justification explaining why this candidate 
is an excellent fit for the Senior AI Engineer role. Be specific and reference 
concrete signals from their profile.

Rules:
1. Exactly 2 sentences, no more, no less.
2. Be specific (reference actual skills, roles, or experiences from their profile).
3. Be persuasive and positive (focus on fit, not weaknesses).
4. Do NOT use generic phrases like "has relevant experience" — be concrete.
5. Do NOT include the candidate name or ID.

Example Output:
"Built large-scale embedding retrieval systems at Google using Pinecone and achieved 15% MRR improvement, demonstrating mastery of the core tech stack. Led cross-functional teams on production ML systems with A/B testing frameworks, aligning perfectly with the team's focus on ship-it culture and ranking systems."

Now generate the reasoning for the candidate above:
"""
    return prompt.strip()


def generate_llm_reasoning(
    candidate: Dict[str, Any],
    jd_text: str,
    fallback_reasoning: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    timeout_seconds: int = 30,
    use_cache: bool = True,
) -> str:
    """
    Generate LLM-powered reasoning for a single candidate.
    
    Args:
        candidate: Candidate profile dictionary
        jd_text: Job description text
        fallback_reasoning: Fallback reasoning if LLM fails. If None, returns deterministic text.
        api_key: LLM API key (optional, can use env variables). 
                 For OpenAI, set OPENAI_API_KEY. For Anthropic, set ANTHROPIC_API_KEY, etc.
        model: Model name (e.g., "gpt-3.5-turbo", "claude-3-sonnet-20240229", "gemini-pro")
        temperature: Sampling temperature (0.0-1.0). Default 0.3 for consistency.
        timeout_seconds: Max time to wait for API response. Default 30s.
        use_cache: Whether to cache results to avoid duplicate API calls. Default True.
    
    Returns:
        2-sentence reasoning string. Falls back gracefully if LLM is unavailable.
    
    Example:
        >>> candidate = {'candidate_id': 'c1', 'profile': {...}, ...}
        >>> jd = "Senior AI Engineer..."
        >>> reasoning = generate_llm_reasoning(candidate, jd)
        >>> print(reasoning)
        "Built embedding systems at Google using Pinecone... Led ML teams with A/B testing..."
    """
    
    if not HAS_LITELLM:
        logger.debug("litellm not available; using fallback reasoning")
        return fallback_reasoning or _get_default_fallback()
    
    # Check cache
    if use_cache:
        jd_hash = _hash_jd(jd_text)
        cache_key = _get_cache_key(candidate.get("candidate_id", "unknown"), jd_hash)
        
        if cache_key in _reasoning_cache:
            logger.debug(f"Cache hit for {cache_key}")
            return _reasoning_cache[cache_key]
    
    try:
        # Construct prompt
        prompt = _construct_llm_prompt(candidate, jd_text)
        
        # Call LLM via litellm (multi-provider abstraction)
        logger.debug(f"Calling LLM ({model}) for candidate {candidate.get('candidate_id', 'unknown')}")
        
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=256,
            api_key=api_key,  # Will use env vars if None
            timeout=timeout_seconds,
            request_timeout=timeout_seconds,
        )
        
        # Extract reasoning from response
        reasoning = response.choices[0].message.content.strip()
        
        # Validate and truncate if needed
        if not reasoning or len(reasoning) < 20:
            logger.warning(
                f"LLM response too short for {candidate.get('candidate_id', 'unknown')}: '{reasoning}'. "
                "Using fallback."
            )
            reasoning = fallback_reasoning or _get_default_fallback()
        
        # Cache result
        if use_cache:
            _reasoning_cache[cache_key] = reasoning
        
        logger.info(
            f"Generated LLM reasoning for {candidate.get('candidate_id', 'unknown')} "
            f"({len(reasoning)} chars)"
        )
        
        return reasoning
    
    except Exception as e:
        # Graceful fallback on any error
        logger.warning(
            f"LLM reasoning generation failed for {candidate.get('candidate_id', 'unknown')}: {e}. "
            "Using fallback reasoning.",
            exc_info=False,  # Avoid verbose stack trace for expected failures
        )
        return fallback_reasoning or _get_default_fallback()


def batch_generate_llm_reasoning(
    candidates: List[Dict[str, Any]],
    jd_text: str,
    fallback_fn=None,
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    timeout_seconds: int = 30,
    max_workers: int = 1,
) -> List[Tuple[str, str]]:
    """
    Generate LLM reasoning for multiple candidates (optionally in parallel).
    
    This is a simple implementation that calls generate_llm_reasoning() for each candidate.
    For true parallel execution, consider using concurrent.futures.ThreadPoolExecutor
    (though API rate limits may apply).
    
    Args:
        candidates: List of candidate dictionaries
        jd_text: Job description
        fallback_fn: Function to call for fallback reasoning. Signature: fallback_fn(candidate, jd_text) -> str
        api_key: LLM API key
        model: Model name
        temperature: Sampling temperature
        timeout_seconds: Timeout per call
        max_workers: Number of parallel workers (currently unused; set to 1)
    
    Returns:
        List of (candidate_id, reasoning) tuples
    
    Example:
        >>> all_reasoning = batch_generate_llm_reasoning(top_100, jd)
        >>> for cid, reasoning in all_reasoning:
        ...     print(f"{cid}: {reasoning}")
    """
    results = []
    
    for i, candidate in enumerate(candidates):
        cid = candidate.get("candidate_id", f"cand_{i}")
        
        # Get fallback reasoning if function provided
        fallback = None
        if fallback_fn:
            try:
                fallback = fallback_fn(candidate, jd_text)
            except Exception as e:
                logger.warning(f"Fallback function failed for {cid}: {e}")
        
        reasoning = generate_llm_reasoning(
            candidate,
            jd_text,
            fallback_reasoning=fallback,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        
        results.append((cid, reasoning))
        
        if (i + 1) % 10 == 0:
            logger.info(f"Generated reasoning for {i + 1}/{len(candidates)} candidates")
    
    logger.info(f"Batch generation complete: {len(candidates)} candidates processed")
    return results


def _get_default_fallback() -> str:
    """Return a generic fallback reasoning when LLM is unavailable."""
    return (
        "Qualified candidate with relevant experience in AI/ML and production systems. "
        "Skills and background align with the role requirements."
    )


def clear_cache() -> None:
    """Clear the in-memory reasoning cache (useful for memory management in long-running processes)."""
    global _reasoning_cache
    size = len(_reasoning_cache)
    _reasoning_cache.clear()
    logger.info(f"Cleared reasoning cache ({size} entries)")


if __name__ == "__main__":
    """
    Quick test: Generate reasoning for a sample candidate.
    Requires OPENAI_API_KEY (or other provider key) to be set in environment.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Mock candidate
    sample_candidate = {
        "candidate_id": "cand_001",
        "profile": {
            "current_title": "Senior ML Engineer",
            "headline": "10+ years ML, LLMs, embeddings, production systems",
            "summary": "Built large-scale retrieval systems and ranking pipelines at Google",
            "years_of_experience": 10,
            "location": "Bangalore, India",
        },
        "skills": [
            {"name": "Python", "proficiency": "advanced"},
            {"name": "PyTorch", "proficiency": "advanced"},
            {"name": "Vector Databases", "proficiency": "advanced"},
            {"name": "LLMs", "proficiency": "intermediate"},
        ],
        "career_history": [
            {
                "title": "Senior ML Engineer",
                "company": "Google",
                "description": "Built embeddings and retrieval systems",
            },
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "notice_period_days": 15,
        },
    }
    
    jd = """
    Senior AI Engineer for embeddings + retrieval ranking systems.
    Required: 5-9 years production ML, vector databases, ranking, LLM fine-tuning.
    Tech stack: sentence-transformers, BGE, Pinecone, Weaviate, XGBoost.
    Must ship working code and have A/B testing experience.
    """
    
    print("Testing generate_llm_reasoning()...")
    print("(Requires OPENAI_API_KEY or ANTHROPIC_API_KEY in environment)")
    print()
    
    reasoning = generate_llm_reasoning(sample_candidate, jd)
    print(f"Generated Reasoning:\n{reasoning}\n")
    
    print("Cache state after 1 call:")
    print(f"  Cache size: {len(_reasoning_cache)}")
    
    # Test cache hit
    reasoning_cached = generate_llm_reasoning(sample_candidate, jd)
    print(f"Cache hit test (should be instant):\n{reasoning_cached}\n")
