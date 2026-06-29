"""
Learning-to-Rank (LTR) Model Training

This script trains an XGBoost classifier to learn optimal feature weights for candidate
ranking. Instead of using hardcoded weights (0.25 * skill + 0.16 * exp + ...), we train
the model on a labeled dataset of Hire/Reject decisions.

Usage:
    python src/train_ltr.py --labeled-data data/raw/labeled_candidates.csv \
                             --output models/xgb_ranker.json

Input CSV Format (data/raw/labeled_candidates.csv):
    candidate_id,label
    cand_001,1
    cand_002,0
    cand_003,1
    ...

Output:
    models/xgb_ranker.json — trained XGBoost model (serialized)

Integration in rank.py:
    - Load model at startup
    - In compute_score(), use model.predict_proba() if model available
    - Fall back to hardcoded weights if model missing or xgboost not installed
"""

import os
import json
import logging
import argparse
import sys
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional import with graceful fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning(
        "xgboost not installed. LTR training disabled. "
        "Install via: pip install xgboost"
    )


def load_rank_module():
    """
    Dynamically load rank.py module to access its scoring functions.
    
    Returns:
        The rank module
    
    Raises:
        ImportError: If rank.py cannot be loaded
    """
    try:
        import importlib.util
        rank_path = os.path.join(os.path.dirname(__file__), "..", "rank.py")
        spec = importlib.util.spec_from_file_location("rank", rank_path)
        rank_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rank_module)
        return rank_module
    except Exception as e:
        logger.error(f"Failed to load rank.py: {e}")
        raise


def extract_features(candidate: Dict[str, Any], rank_module) -> np.ndarray:
    """
    Extract the 11 scoring features from a candidate using rank.py's scoring functions.
    
    Features (in order):
    0. skill_score
    1. experience_score
    2. production_score
    3. behavioral_score
    4. location_score
    5. title_relevance_score
    6. assessment_quality_score
    7. edu_tier_score
    8. cert_score
    9. notice_period_score
    10. consulting_penalty
    
    Args:
        candidate: Candidate dictionary
        rank_module: Imported rank module
    
    Returns:
        Numpy array of 11 features
    """
    try:
        features = [
            rank_module.skill_score(candidate),
            rank_module.experience_score(candidate),
            rank_module.production_score(candidate),
            rank_module.behavioral_score(candidate),
            rank_module.location_score(candidate),
            rank_module.title_relevance_score(candidate),
            rank_module.assessment_quality_score(candidate),
            rank_module.edu_tier_score(candidate),
            rank_module.cert_score(candidate),
            rank_module.notice_period_score(candidate),
            rank_module.consulting_penalty(candidate),
        ]
        
        # Handle None values (especially assessment_quality_score which can be None)
        features = [f if f is not None else 0.0 for f in features]
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        logger.error(f"Failed to extract features for candidate {candidate.get('candidate_id')}: {e}")
        raise


def load_labeled_candidates(
    csv_path: str,
    candidates_jsonl_path: str = "candidates.jsonl",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Load labeled candidates from CSV and match with full profiles from JSONL.
    
    Args:
        csv_path: Path to labeled CSV (candidate_id, label)
        candidates_jsonl_path: Path to full candidates.jsonl
    
    Returns:
        Tuple of (labels_df, candidate_profiles_dict)
    """
    # Load labels
    labels_df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(labels_df)} labeled candidates from {csv_path}")
    
    # Load candidate profiles
    candidates = {}
    with open(candidates_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                candidates[c['candidate_id']] = c
    
    logger.info(f"Loaded {len(candidates)} candidate profiles from {candidates_jsonl_path}")
    
    return labels_df, candidates


def build_training_data(
    labels_df: pd.DataFrame,
    candidates: Dict[str, Dict[str, Any]],
    rank_module,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and target vector y.
    
    Args:
        labels_df: DataFrame with candidate_id and label columns
        candidates: Dictionary of candidate profiles
        rank_module: Imported rank module
    
    Returns:
        Tuple of (X, y) where X is (N, 11) and y is (N,)
    """
    X_list = []
    y_list = []
    
    for idx, row in labels_df.iterrows():
        cid = row['candidate_id']
        label = int(row['label'])
        
        if cid not in candidates:
            logger.warning(f"Candidate {cid} not found in candidates.jsonl, skipping")
            continue
        
        candidate = candidates[cid]
        
        try:
            features = extract_features(candidate, rank_module)
            X_list.append(features)
            y_list.append(label)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"  Processed {idx + 1}/{len(labels_df)} labeled candidates")
        
        except Exception as e:
            logger.warning(f"Failed to extract features for {cid}: {e}")
            continue
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    logger.info(f"Built training data: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"  Class distribution: {np.bincount(y)}")
    
    return X, y


def train_xgb_model(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier for ranking.
    
    Args:
        X: Feature matrix (N, 11)
        y: Target vector (N,) with values 0 or 1
        n_estimators: Number of boosting rounds
        max_depth: Max tree depth
        learning_rate: Learning rate
    
    Returns:
        Trained XGBClassifier model
    """
    logger.info(f"Training XGBoost model (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})...")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
    )
    
    model.fit(X, y)
    
    logger.info("Training complete!")
    
    # Log feature importance
    importance = model.feature_importances_
    feature_names = [
        "skill", "experience", "production", "behavioral", "location",
        "title", "assessment", "education", "certification", "notice", "consulting_penalty"
    ]
    
    logger.info("Feature Importance:")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        logger.info(f"  {name:20s}: {imp:.4f}")
    
    return model


def save_model(model: xgb.XGBClassifier, output_path: str) -> None:
    """
    Save XGBoost model to JSON file.
    
    Args:
        model: Trained XGBClassifier
        output_path: Path to save model (e.g., models/xgb_ranker.json)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)
    logger.info(f"Saved model to {output_path}")


def evaluate_model(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model on training data (for feedback).
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
    
    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_pred = model.predict(X)
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
    }
    
    logger.info("Training Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost Learning-to-Rank model"
    )
    parser.add_argument(
        "--labeled-data",
        type=str,
        default="data/raw/labeled_candidates.csv",
        help="Path to labeled candidates CSV (candidate_id, label)",
    )
    parser.add_argument(
        "--candidates-jsonl",
        type=str,
        default="candidates.jsonl",
        help="Path to candidates JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/xgb_ranker.json",
        help="Output model path",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting rounds",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Max tree depth",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        # Check xgboost
        if not HAS_XGBOOST:
            raise RuntimeError(
                "xgboost not installed. Install via: pip install xgboost"
            )
        
        # Load rank module
        logger.info("Loading rank.py module...")
        rank_module = load_rank_module()
        
        # Load labeled data
        if not os.path.exists(args.labeled_data):
            raise FileNotFoundError(f"Labeled data file not found: {args.labeled_data}")
        
        labels_df, candidates = load_labeled_candidates(
            args.labeled_data,
            args.candidates_jsonl,
        )
        
        # Build training data
        X, y = build_training_data(labels_df, candidates, rank_module)
        
        if len(X) < 10:
            raise ValueError(f"Not enough training samples: {len(X)}")
        
        # Train model
        model = train_xgb_model(
            X, y,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
        
        # Evaluate
        evaluate_model(model, X, y)
        
        # Save model
        save_model(model, args.output)
        
        logger.info(f"✓ LTR training complete!")
        logger.info(f"  Model saved to: {args.output}")
        logger.info(f"  Training samples: {len(X)}")
        logger.info(f"  Features per sample: {X.shape[1]}")
        
        return 0
    
    except Exception as e:
        logger.error(f"LTR training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
