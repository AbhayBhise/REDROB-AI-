#!/usr/bin/env python3
"""
End-to-End Pipeline Runner & Validator

Runs all offline preprocessing steps and the final ranking pipeline,
then validates the output submission.csv.

Usage:
    python scripts/run_pipeline.py --full        # Run everything
    python scripts/run_pipeline.py --rank-only   # Skip preprocessing
    python scripts/run_pipeline.py --validate    # Validate only
"""

import os
import sys
import subprocess
import argparse
import json
import csv
from datetime import datetime

def run_command(cmd, description, critical=True):
    """Run a shell command and log output."""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        msg = f"\n❌ FAILED: {description}"
        if critical:
            print(msg)
            sys.exit(1)
        else:
            print(f"\n⚠️  WARNING: {description} (non-critical, continuing...)")
    else:
        print(f"\n✅ SUCCESS: {description}")
    
    return result.returncode == 0

def validate_submission(submission_path):
    """Validate submission.csv format and content."""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Validating submission.csv")
    print(f"{'='*70}\n")
    
    if not os.path.exists(submission_path):
        print(f"❌ ERROR: {submission_path} not found!")
        return False
    
    try:
        with open(submission_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Check header
        expected_headers = {'rank', 'candidate_id', 'score', 'reasoning'}
        actual_headers = set(reader.fieldnames or [])
        
        if not expected_headers.issubset(actual_headers):
            print(f"❌ ERROR: Missing required columns. Expected: {expected_headers}, Got: {actual_headers}")
            return False
        
        # Check row count
        if len(rows) != 100:
            print(f"⚠️  WARNING: Expected 100 rows, got {len(rows)}")
        else:
            print(f"✅ Row count: {len(rows)} (correct)")
        
        # Check scores
        scores = []
        for i, row in enumerate(rows):
            try:
                rank = int(row.get('rank', i+1))
                score = float(row.get('score', 0))
                reasoning = row.get('reasoning', '')
                cid = row.get('candidate_id', '')
                
                if rank != i + 1:
                    print(f"⚠️  WARNING: Row {i+1} has rank={rank}, expected {i+1}")
                
                if not (0 <= score <= 1):
                    print(f"⚠️  WARNING: Row {i+1} has score={score}, expected [0, 1]")
                
                if not cid:
                    print(f"❌ ERROR: Row {i+1} missing candidate_id")
                    return False
                
                if not reasoning or len(reasoning) < 10:
                    print(f"⚠️  WARNING: Row {i+1} has very short reasoning")
                
                scores.append(score)
            
            except ValueError as e:
                print(f"❌ ERROR: Row {i+1} has invalid values: {e}")
                return False
        
        # Check score ordering
        scores_sorted = sorted(scores, reverse=True)
        if scores == scores_sorted:
            print(f"✅ Scores are sorted in descending order")
        else:
            print(f"⚠️  WARNING: Scores are not in descending order")
        
        # Print summary
        print(f"\n✅ Sample rows (top 3):")
        for i in range(min(3, len(rows))):
            row = rows[i]
            print(f"  Rank {row['rank']}: {row['candidate_id']} (score={row['score']}, reasoning_len={len(row['reasoning'])})")
        
        print(f"\n✅ Validation PASSED!")
        return True
    
    except Exception as e:
        print(f"❌ ERROR: Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline runner & validator")
    parser.add_argument('--full', action='store_true', help='Run all steps (preprocessing + ranking)')
    parser.add_argument('--rank-only', action='store_true', help='Skip preprocessing, run ranking only')
    parser.add_argument('--validate', action='store_true', help='Validate submission.csv only')
    parser.add_argument('--skip-expand-jd', action='store_true', help='Skip JD expansion (faster)')
    parser.add_argument('--skip-ltr', action='store_true', help='Skip LTR training (faster)')
    
    args = parser.parse_args()
    
    # Default to --full if no args specified
    if not any([args.full, args.rank_only, args.validate]):
        args.full = True
    
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(root_dir)
    
    print(f"\n{'='*70}")
    print(f"🚀 Redrob AI — End-to-End Pipeline Runner")
    print(f"{'='*70}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Step 1: Expand JD (optional, offline)
    if args.full and not args.skip_expand_jd:
        run_command(
            [sys.executable, 'src/expand_jd.py'],
            'Step 1: LLM JD Expansion',
            critical=False  # Optional
        )
    
    # Step 2: Train LTR model (optional, offline)
    if args.full and not args.skip_ltr:
        if os.path.exists('data/raw/labeled_candidates.csv'):
            run_command(
                [sys.executable, 'src/train_ltr.py', 
                 '--labeled-data', 'data/raw/labeled_candidates.csv',
                 '--output', 'models/xgb_ranker.json'],
                'Step 2: XGBoost LTR Training',
                critical=False  # Optional
            )
        else:
            print(f"\n⚠️  Skipping LTR training: labeled_candidates.csv not found")
    
    # Step 3: Precompute embeddings (always needed unless rank-only)
    if args.full or (not args.rank_only and not args.validate):
        run_command(
            [sys.executable, 'precompute.py'],
            'Step 3: Precompute Embeddings (bge-base, 768-dim, compressed)',
            critical=True
        )
    
    # Step 4: Run ranking pipeline
    if args.full or args.rank_only:
        run_command(
            [sys.executable, 'rank.py',
             '--candidates', 'candidates.jsonl',
             '--out', 'submission.csv'],
            'Step 4: Final Ranking Pipeline',
            critical=True
        )
    
    # Step 5: Validate submission
    if args.full or args.rank_only or args.validate:
        success = validate_submission('submission.csv')
        if not success:
            sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"🎉 Pipeline Complete!")
    print(f"{'='*70}")
    print(f"\nOutput: submission.csv")
    print(f"Format: rank, candidate_id, score, reasoning")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\n✅ Ready for submission!")

if __name__ == '__main__':
    main()
