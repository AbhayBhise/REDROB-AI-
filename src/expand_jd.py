"""
================================================================================
WARNING: EXPERIMENTAL FEATURE
================================================================================
This script is strictly experimental and NOT part of the official Stage 3 offline
evaluation pipeline. 

The primary Stage 3 ranking engine (rank.py) relies entirely on a fully 
offline, deterministic hybrid retrieval module. 
This file is kept only for archival purposes. 
Do not use this for formal offline evaluation as it requires external API access.
================================================================================

LLM Query & Job Description Expansion

This is an OFFLINE utility script that runs once (not during ranking).
It expands the JD text with synonyms, related frameworks, and alternate terminologies
to improve BM25 lexical recall. For example, if the JD mentions "Vector Databases",
this script generates ["ChromaDB", "Vespa", "pgvector", "Weaviate", ...].

Usage:
    python src/expand_jd.py --jd-text "Senior AI Engineer..." --output data/processed/expanded_keywords.json
    python src/expand_jd.py --load-from-rank-py --output data/processed/expanded_keywords.json

Output:
    JSON file with array of keyword strings: ["keyword1", "keyword2", ...]

Integration in rank.py:
    - Load expanded keywords from data/processed/expanded_keywords.json (if exists)
    - Append to BM25 tokenized JD (repeat 2x for weight)
    - If file missing, silently continue with original JD
"""

import json
import logging
import os
import argparse
from typing import List, Optional

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    logger.warning(
        "litellm not installed. JD expansion disabled. "
        "Install via: pip install litellm"
    )


def get_jd_from_rank_py() -> str:
    """
    Load the JD_TEXT constant from rank.py.
    This avoids hardcoding the JD in multiple files.
    """
    try:
        import sys
        rank_path = os.path.join(os.path.dirname(__file__), "..", "rank.py")
        spec = __import__("importlib.util").util.spec_from_file_location("rank", rank_path)
        rank_module = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(rank_module)
        
        if hasattr(rank_module, "JD_TEXT"):
            return rank_module.JD_TEXT
        else:
            raise AttributeError("JD_TEXT not found in rank.py")
    except Exception as e:
        logger.error(f"Failed to load JD_TEXT from rank.py: {e}")
        raise


def expand_jd_with_llm(
    jd_text: str,
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    timeout_seconds: int = 60,
) -> List[str]:
    """
    Use an LLM to generate a list of expanded keywords and alternate terminologies
    based on the job description.
    
    Args:
        jd_text: Original job description text
        api_key: LLM API key (optional; uses environment variables if not provided)
        model: Model name (e.g., "gpt-3.5-turbo", "claude-3-sonnet-20240229")
        temperature: Sampling temperature for creativity (default 0.3 for consistency)
        timeout_seconds: Max time to wait for API response
    
    Returns:
        List of expanded keywords (e.g., ["chromadb", "vespa", "pgvector", ...])
    
    Raises:
        RuntimeError: If litellm is not installed or LLM call fails
    """
    if not HAS_LITELLM:
        raise RuntimeError(
            "litellm not installed. Install via: pip install litellm"
        )
    
    prompt = f"""
You are an expert in AI/ML recruiting and technical taxonomy. Your task is to expand 
a job description with synonyms, related frameworks, alternative terminologies, and 
related concepts that highly-skilled candidates might use in their profiles.

Original Job Description:
{jd_text}

Task:
Generate a comprehensive list of technical keywords, frameworks, and related terms that 
candidates for this role might mention in their profiles. Include:
1. Alternative names for technologies (e.g., "ChromaDB" for "Vector Databases")
2. Related frameworks and libraries (e.g., "LangChain", "Llama Index" for LLM work)
3. Synonyms for key concepts (e.g., "semantic search", "dense retrieval", "approximate nearest neighbor")
4. Related programming languages and tools
5. Alternative job titles or role names that fit this profile

Rules:
- Return ONLY a valid JSON array of strings (lowercase, no extra text)
- Each keyword should be a single term or short phrase (2-3 words max)
- Include both broad and specific terms
- Aim for 50-100 keywords total
- Do NOT include the original JD text or any commentary

Example format:
["chromadb", "vespa", "pgvector", "weaviate", "pinecone", "qdrant", "langchain", "llama-index", "semantic search", "approximate nearest neighbor", "dense retrieval", "sparse retrieval", "bm25", "vector embedding", "representation learning", "neural information retrieval"]

Now generate the expanded keyword list for the above JD:
"""
    
    try:
        logger.info(f"Calling LLM ({model}) to expand JD...")
        
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000,
            api_key=api_key,
            timeout=timeout_seconds,
            request_timeout=timeout_seconds,
        )
        
        # Extract response
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        try:
            keywords = json.loads(response_text)
            if not isinstance(keywords, list):
                raise ValueError("Response is not a JSON array")
            
            # Clean keywords (lowercase, strip whitespace)
            keywords = [k.strip().lower() for k in keywords if k.strip()]
            
            logger.info(f"Generated {len(keywords)} expanded keywords")
            return keywords
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {response_text}\nError: {e}")
            raise ValueError(f"LLM response is not valid JSON: {e}")
    
    except Exception as e:
        logger.error(f"LLM expansion failed: {e}", exc_info=True)
        raise


def save_expanded_keywords(keywords: List[str], output_path: str) -> None:
    """
    Save expanded keywords to a JSON file.
    
    Args:
        keywords: List of keyword strings
        output_path: Path to save JSON file (e.g., "data/processed/expanded_keywords.json")
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(keywords, f, indent=2)
    
    logger.info(f"Saved {len(keywords)} keywords to {output_path}")


def main():
    """
    Main entry point for the expansion script.
    """
    parser = argparse.ArgumentParser(
        description="Expand JD with LLM-generated keywords for improved BM25 recall"
    )
    parser.add_argument(
        "--jd-text",
        type=str,
        default=None,
        help="Job description text (if not provided, loads from rank.py)",
    )
    parser.add_argument(
        "--load-from-rank-py",
        action="store_true",
        help="Load JD_TEXT from rank.py (default behavior)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/expanded_keywords.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="LLM model name",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API key (uses environment variable if not provided)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM sampling temperature (0.0-1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="API call timeout in seconds",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        # Get JD text
        if args.jd_text:
            jd_text = args.jd_text
            logger.info("Using provided JD text")
        else:
            jd_text = get_jd_from_rank_py()
            logger.info("Loaded JD_TEXT from rank.py")
        
        # Expand JD with LLM
        keywords = expand_jd_with_llm(
            jd_text,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
        )
        
        # Save to disk
        save_expanded_keywords(keywords, args.output)
        
        logger.info(f"✓ JD expansion complete!")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Keywords: {len(keywords)}")
        
        # Print sample
        print("\nSample expanded keywords:")
        print(json.dumps(keywords[:10], indent=2))
        print(f"... and {len(keywords) - 10} more")
    
    except Exception as e:
        logger.error(f"JD expansion failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
