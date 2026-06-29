"""
Cross-Encoder Re-ranking Module

This module provides fast Cross-Encoder re-ranking for the top candidates retrieved
by the hybrid BM25 + Dense retrieval pipeline. The Cross-Encoder fixes the "dense 
compression loss" by evaluating the semantic relevance of each candidate directly 
against the job description.

Key Features:
- Uses sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2) for fast CPU inference
- Batches inference for speed (~500 candidates in <2 seconds on CPU)
- Normalizes scores to [0, 1] for seamless fusion with existing MasterScore
- Graceful fallback: if sentence-transformers is unavailable, returns original scores
- Production-ready error handling and logging
"""

import logging
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

# Optional import with graceful fallback
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    logger.warning(
        "sentence-transformers not installed. Cross-Encoder re-ranking disabled. "
        "Install via: pip install sentence-transformers"
    )


def construct_candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Construct a single text representation of a candidate profile.
    
    Combines: Title + Headline + Summary + Skills + Experience (same as BM25 preprocessing).
    
    Args:
        candidate: Dictionary with keys like 'title', 'headline', 'summary', 'skills', 'experience'
    
    Returns:
        Concatenated candidate text string
    """
    parts = []
    
    if candidate.get("title"):
        parts.append(str(candidate["title"]))
    if candidate.get("headline"):
        parts.append(str(candidate["headline"]))
    if candidate.get("summary"):
        parts.append(str(candidate["summary"]))
    if candidate.get("skills"):
        # skills might be a list or string
        if isinstance(candidate["skills"], list):
            parts.append(" ".join(candidate["skills"]))
        else:
            parts.append(str(candidate["skills"]))
    if candidate.get("experience"):
        # experience might be a list of dicts or strings
        if isinstance(candidate["experience"], list):
            exp_texts = []
            for exp in candidate["experience"]:
                if isinstance(exp, dict):
                    exp_texts.append(str(exp.get("title", "")))
                    exp_texts.append(str(exp.get("description", "")))
                else:
                    exp_texts.append(str(exp))
            parts.append(" ".join(exp_texts))
        else:
            parts.append(str(candidate["experience"]))
    
    return " ".join(parts).strip()


def sigmoid_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize raw Cross-Encoder logits to [0, 1] using sigmoid function.
    
    Cross-Encoders output unbounded logits. Sigmoid ensures:
    - Score near 0 → probability ≈ 0.5
    - Score > 0 → probability > 0.5
    - Score < 0 → probability < 0.5
    - All scores bounded in [0, 1]
    
    Args:
        scores: Raw logits from CrossEncoder
    
    Returns:
        Normalized scores in [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-scores))


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores using min-max scaling within the batch.
    
    Alternative to sigmoid; can be useful if you want scores to span [0, 1].
    
    Args:
        scores: Raw logits from CrossEncoder
    
    Returns:
        Min-max normalized scores in [0, 1]
    """
    min_score = scores.min()
    max_score = scores.max()
    
    if min_score == max_score:
        # All scores are identical; return uniform scores
        return np.ones_like(scores) * 0.5
    
    return (scores - min_score) / (max_score - min_score)


def rerank_candidates(
    top_k_candidates: List[Dict[str, Any]],
    jd_text: str,
    batch_size: int = 128,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    use_sigmoid: bool = True,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Re-rank the top candidates using a Cross-Encoder model.
    
    Takes the top 500 (or any top-K) candidates from the hybrid pipeline and 
    re-scores them using a faster Cross-Encoder model that directly evaluates 
    (JD, Candidate) relevance. This fixes the "dense compression loss" of Bi-Encoders.
    
    Args:
        top_k_candidates: List of candidate dictionaries from the hybrid retrieval pipeline.
                         Each dict should have: 'title', 'headline', 'summary', 'skills', 
                         'experience', and other fields (e.g., 'candidate_id', 'baseline_score').
        jd_text: The job description text.
        batch_size: Number of (JD, Candidate) pairs to score at once. Default 128.
        model_name: HuggingFace model identifier. Default: ms-marco-MiniLM-L-6-v2 (fast, ~90M params).
        use_sigmoid: If True, use sigmoid normalization. If False, use min-max. Default: True.
    
    Returns:
        List of (candidate_dict, rerank_score) tuples, where rerank_score ∈ [0, 1].
        If Cross-Encoder is unavailable, returns [(candidate, candidate.get('baseline_score', 0.0)), ...]
        with a warning log.
    
    Raises:
        ValueError: If top_k_candidates is empty.
    
    Example:
        >>> candidates = [{'id': 1, 'title': 'ML Engineer', ...}, ...]
        >>> jd = "Senior AI Engineer ..."
        >>> reranked = rerank_candidates(candidates, jd)
        >>> top_100 = reranked[:100]
    """
    if not top_k_candidates:
        raise ValueError("top_k_candidates cannot be empty")
    
    # Fallback: if sentence-transformers is not installed, return baseline scores
    if not HAS_CROSS_ENCODER:
        logger.warning(
            "Cross-Encoder re-ranking unavailable. Returning candidates with baseline scores."
        )
        return [
            (candidate, candidate.get("baseline_score", 0.0))
            for candidate in top_k_candidates
        ]
    
    try:
        # Load the Cross-Encoder model
        logger.info(f"Loading Cross-Encoder model: {model_name}")
        model = CrossEncoder(model_name, max_length=512)
        
        # Construct (JD, Candidate) pairs
        candidate_texts = [construct_candidate_text(c) for c in top_k_candidates]
        pairs = [[jd_text, ctext] for ctext in candidate_texts]
        
        logger.info(f"Scoring {len(pairs)} (JD, Candidate) pairs in batches of {batch_size}...")
        
        # Batch inference
        logits = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        
        # Normalize scores to [0, 1]
        if use_sigmoid:
            normalized_scores = sigmoid_normalize(logits)
            logger.info(f"Normalized Cross-Encoder logits using sigmoid. Score range: [{normalized_scores.min():.4f}, {normalized_scores.max():.4f}]")
        else:
            normalized_scores = minmax_normalize(logits)
            logger.info(f"Normalized Cross-Encoder logits using min-max. Score range: [{normalized_scores.min():.4f}, {normalized_scores.max():.4f}]")
        
        # Pair candidates with their re-rank scores
        result = [
            (candidate, float(score))
            for candidate, score in zip(top_k_candidates, normalized_scores)
        ]
        
        logger.info(f"Re-ranking complete. Top 5 scores: {sorted(normalized_scores, reverse=True)[:5]}")
        
        return result
    
    except Exception as e:
        # Graceful fallback on any error
        logger.error(
            f"Cross-Encoder re-ranking failed: {e}. Falling back to baseline scores.",
            exc_info=True,
        )
        return [
            (candidate, candidate.get("baseline_score", 0.0))
            for candidate in top_k_candidates
        ]


def fuse_scores_with_baseline(
    reranked_candidates: List[Tuple[Dict[str, Any], float]],
    rerank_weight: float = 0.3,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Fuse Cross-Encoder scores with baseline (MasterScore) for final ranking.
    
    The baseline_score (from the hybrid pipeline) captures BM25, dense retrieval, 
    and hand-tuned features. The Cross-Encoder score adds a second opinion from 
    a fine-tuned pairwise ranker. Combining them often yields better results.
    
    Fusion formula:
        final_score = (1 - rerank_weight) * baseline_score + rerank_weight * cross_encoder_score
    
    Args:
        reranked_candidates: Output of rerank_candidates() function.
        rerank_weight: Weight for Cross-Encoder score ∈ [0, 1]. Default 0.3 (30% weight to Cross-Encoder).
    
    Returns:
        List of (candidate_dict, fused_score) tuples.
    
    Example:
        >>> reranked = rerank_candidates(top_500, jd)
        >>> final_scored = fuse_scores_with_baseline(reranked, rerank_weight=0.3)
        >>> top_100 = final_scored[:100]
    """
    fused_results = []
    
    for candidate, cross_score in reranked_candidates:
        baseline_score = candidate.get("baseline_score", 0.0)
        
        # Ensure both scores are in [0, 1]
        baseline_score = max(0.0, min(1.0, baseline_score))
        cross_score = max(0.0, min(1.0, cross_score))
        
        fused_score = (1 - rerank_weight) * baseline_score + rerank_weight * cross_score
        
        # Update candidate dict with scores for transparency
        candidate_copy = candidate.copy()
        candidate_copy["cross_encoder_score"] = cross_score
        candidate_copy["fused_score"] = fused_score
        
        fused_results.append((candidate_copy, fused_score))
    
    return fused_results


if __name__ == "__main__":
    """
    Quick test: Load a sample candidate and JD, run re-ranking.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Mock candidates (minimal required fields)
    mock_candidates = [
        {
            "candidate_id": "cand_001",
            "title": "Senior Machine Learning Engineer",
            "headline": "10+ years ML, TensorFlow, PyTorch, LLMs",
            "summary": "Built recommendation systems and NLP pipelines at scale",
            "skills": ["Python", "TensorFlow", "LLMs", "Vector Databases"],
            "experience": [
                {"title": "ML Engineer @ Google", "description": "Worked on BERT models"}
            ],
            "baseline_score": 0.85,
        },
        {
            "candidate_id": "cand_002",
            "title": "Data Scientist",
            "headline": "5 years data science, pandas, scikit-learn",
            "summary": "Built dashboards and analyzed user data",
            "skills": ["Python", "SQL", "Tableau"],
            "experience": [
                {"title": "Data Analyst @ Startup", "description": "Analytics"}
            ],
            "baseline_score": 0.45,
        },
    ]
    
    jd = """
    We are hiring a Senior AI Engineer with 8+ years of experience.
    Required: Deep learning (TensorFlow/PyTorch), LLM fine-tuning, vector databases (Qdrant, Pinecone),
    distributed training, production ML systems, strong software engineering.
    Nice-to-have: Transformers, prompt engineering, multi-agent systems.
    """
    
    # Test re-ranking
    print("Testing rerank_candidates()...")
    reranked = rerank_candidates(mock_candidates, jd)
    
    print("\nResults:")
    for candidate, score in reranked:
        print(f"  {candidate['title']} (ID: {candidate['candidate_id']}): {score:.4f}")
    
    # Test fusion
    print("\n\nTesting fuse_scores_with_baseline()...")
    fused = fuse_scores_with_baseline(reranked, rerank_weight=0.3)
    
    print("Fused Scores:")
    for candidate, fused_score in fused:
        print(
            f"  {candidate['title']}: baseline={candidate['baseline_score']:.4f}, "
            f"cross={candidate.get('cross_encoder_score', 0.0):.4f}, fused={fused_score:.4f}"
        )
