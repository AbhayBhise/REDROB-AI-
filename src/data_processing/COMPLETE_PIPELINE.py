"""
═══════════════════════════════════════════════════════════════════════════════
INDIA RUNS HACKATHON — TRACK 1 DATA & AI CHALLENGE
Complete End-to-End Pipeline: EDA → Encoding → Feature Engineering → 
Correlation Analysis → Train/Test Split → Model Training & Evaluation
═══════════════════════════════════════════════════════════════════════════════

Team: 4 people (P1: Data Eng, P2: ML/NLP, P3: LLM/RAG, P4: Backend)
Dataset: sample_candidates.json (50 records) or candidates.jsonl (full)
Output: 
  - features_final.csv (all candidates with 80+ features, 0 nulls)
  - train.csv, val.csv, test.csv (70/15/15 split)
  - 10 visualization PNG files
  - Model predictions + performance metrics

Run: python3 COMPLETE_PIPELINE.py
"""

import json
import warnings
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ML
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Stats
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
ROOT_DIR = Path(__file__).resolve().parents[2]
TODAY = datetime(2026, 6, 22)
DATA_DIR = ROOT_DIR / "data" / "external" / "Dataset" / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PLOTS_DIR = PROCESSED_DIR / "plots"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style (dark mode)
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor':'#0f1117',
    'axes.facecolor':'#1a1d27',
    'axes.edgecolor':'#2d3147',
    'text.color':'#e8eaf0',
    'axes.labelcolor':'#c0c8e8',
    'xtick.color':'#8892b0',
    'ytick.color':'#8892b0',
    'grid.color':'#2d3147',
    'grid.alpha':0.5,
    'font.family':'DejaVu Sans',
    'font.size':9,
    'axes.titlesize':11,
    'axes.titleweight':'bold',
    'legend.facecolor':'#1a1d27',
    'legend.edgecolor':'#2d3147',
    'legend.labelcolor':'#c0c8e8'
})

# Color palette
COLORS = ['#6C63FF','#00D4AA','#FF6B6B','#FFD166','#06C8F8','#FF8C42','#A8DADC','#E63946']

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD & FLATTEN DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("SECTION 1: LOADING & FLATTENING DATA")
print("="*80)

with open(DATA_DIR / "sample_candidates.json") as f:
    raw_candidates = json.load(f)

print(f"✓ Loaded {len(raw_candidates)} candidate records")

records = []
for candidate in raw_candidates:
    record = {"candidate_id": candidate["candidate_id"]}
    
    # Profile fields
    record.update(candidate["profile"])
    
    # Career history: aggregate metrics
    career = candidate["career_history"]
    record["num_jobs"] = len(career)
    record["total_career_months"] = sum(x.get("duration_months", 0) or 0 for x in career)
    current_jobs = [x for x in career if x.get("is_current")]
    record["current_tenure_months"] = current_jobs[0]["duration_months"] if current_jobs else 0
    record["has_current_job"] = int(bool(current_jobs))
    record["num_industries"] = len(set(x.get("industry", "") for x in career))
    record["avg_tenure_months"] = (
        record["total_career_months"] / record["num_jobs"] if record["num_jobs"] else 0
    )
    # Seniority: count senior-level roles
    job_titles = [x.get("title", "").lower() for x in career]
    senior_keywords = ["senior", "lead", "principal", "head", "manager", "director", "architect", "vp", "chief"]
    record["senior_roles_count"] = sum(1 for t in job_titles if any(kw in t for kw in senior_keywords))
    
    # Education: tier aggregation
    education = candidate["education"]
    record["num_degrees"] = len(education)
    edu_tiers = [e.get("tier", "") for e in education]
    record["highest_edu_tier"] = edu_tiers[0] if edu_tiers else None
    record["has_tier1_edu"] = int(any("tier_1" in t for t in edu_tiers))
    
    # Skills: proficiency levels & endorsements
    skills = candidate["skills"]
    record["num_skills"] = len(skills)
    record["advanced_skills"] = sum(1 for s in skills if s.get("proficiency") == "advanced")
    record["intermediate_skills"] = sum(1 for s in skills if s.get("proficiency") == "intermediate")
    record["beginner_skills"] = sum(1 for s in skills if s.get("proficiency") == "beginner")
    record["total_endorsements"] = sum(s.get("endorsements", 0) for s in skills)
    record["avg_skill_duration_months"] = (
        np.mean([s.get("duration_months", 0) for s in skills]) if skills else 0
    )
    
    # Certifications
    record["num_certifications"] = len(candidate.get("certifications", []))
    
    # Languages
    languages = candidate.get("languages", [])
    record["num_languages"] = len(languages)
    record["speaks_english"] = int(any(l["language"] == "English" for l in languages))
    
    # Redrob signals (all the recruitment platform engagement metrics)
    signals = candidate["redrob_signals"]
    for key, value in signals.items():
        if key not in ("expected_salary_range_inr_lpa", "skill_assessment_scores"):
            record[key] = value
    
    # Salary range unpacking
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    record["salary_min"] = salary_range.get("min")
    record["salary_max"] = salary_range.get("max")
    record["salary_mid"] = (
        (salary_range.get("min", 0) + salary_range.get("max", 0)) / 2 
        if salary_range else None
    )
    
    # Skill assessment scores
    assessments = signals.get("skill_assessment_scores", {})
    record["num_assessments"] = len(assessments)
    record["avg_assessment_score"] = (
        np.mean(list(assessments.values())) if assessments else None
    )
    
    records.append(record)

df = pd.DataFrame(records)
print(f"✓ Flattened shape: {df.shape}")
print(f"  Columns: {df.shape[1]}")
print(f"  Null values: {df.isnull().sum().sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA QUALITY AUDIT & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 2: DATA QUALITY AUDIT & CLEANING")
print("="*80)

# Parse dates
df["signup_date"] = pd.to_datetime(df["signup_date"])
df["last_active_date"] = pd.to_datetime(df["last_active_date"])
df["days_since_active"] = (TODAY - df["last_active_date"]).dt.days
df["days_on_platform"] = (TODAY - df["signup_date"]).dt.days

# Handle sentinel -1 values (platform encoding for "not available")
print("\n✓ SENTINEL VALUE HANDLING:")
github_sentinel = (df["github_activity_score"] == -1).sum()
print(f"  - github_activity_score = -1: {github_sentinel}/50 (no GitHub account linked)")
df["github_activity_score"] = df["github_activity_score"].clip(lower=0)

offer_sentinel = (df["offer_acceptance_rate"] == -1).sum()
print(f"  - offer_acceptance_rate = -1: {offer_sentinel}/50 (never received offer)")
df["offer_acceptance_rate"] = df["offer_acceptance_rate"].clip(lower=0)

# Handle null assessment scores
assess_nulls = df["avg_assessment_score"].isna().sum()
print(f"  - avg_assessment_score = null: {assess_nulls}/50 (no assessments taken)")
df["avg_assessment_score"] = df["avg_assessment_score"].fillna(0)

# Binary features: ensure 0/1 type
binary_cols = [
    "open_to_work_flag", "willing_to_relocate", "has_current_job",
    "has_tier1_edu", "speaks_english", "verified_email",
    "verified_phone", "linkedin_connected"
]
for col in binary_cols:
    df[col] = df[col].astype(int)

# GitHub active binary
df["github_active"] = (df["github_activity_score"] > 0).astype(int)

print(f"\n✓ AFTER CLEANING: {df.isnull().sum().sum()} remaining nulls")
print(f"  All {len(binary_cols)} binary features encoded as 0/1")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ENCODING (Ordinal, Label, One-Hot)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 3: FEATURE ENCODING")
print("="*80)

# 3A) ORDINAL ENCODING (preserve ordering)
print("\n✓ ORDINAL ENCODING:")

edu_tier_map = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1}
df["edu_tier_enc"] = df["highest_edu_tier"].map(edu_tier_map).fillna(1).astype(int)
print(f"  - Education tier: {edu_tier_map}")

company_size_map = {
    "1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4,
    "501-1000": 5, "1001-5000": 6, "5001-10000": 7, "10001+": 8
}
df["company_size_enc"] = df["current_company_size"].map(company_size_map)
print(f"  - Company size: 1(smallest)→8(largest)")

work_mode_map = {"onsite": 1, "hybrid": 2, "flexible": 3, "remote": 4}
df["work_mode_enc"] = df["preferred_work_mode"].map(work_mode_map)
print(f"  - Work mode: onsite→hybrid→flexible→remote")

# 3B) LABEL ENCODING (no ordering)
print("\n✓ LABEL ENCODING (categorical → integers):")

le_country = LabelEncoder()
df["country_enc"] = le_country.fit_transform(df["country"])
print(f"  - Country: {len(le_country.classes_)} unique values")

le_industry = LabelEncoder()
df["industry_enc"] = le_industry.fit_transform(df["current_industry"])
print(f"  - Industry: {len(le_industry.classes_)} unique values")

# 3C) ONE-HOT ENCODING (for ML models that need binary indicators)
print("\n✓ ONE-HOT ENCODING:")

mode_dummies = pd.get_dummies(df["preferred_work_mode"], prefix="mode_", drop_first=False)
df = pd.concat([df, mode_dummies], axis=1)
print(f"  - preferred_work_mode → {list(mode_dummies.columns)}")

print(f"\n✓ ENCODING COMPLETE: Shape now {df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FEATURE ENGINEERING (Composite Scores)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 4: FEATURE ENGINEERING (Composite Scores)")
print("="*80)

print("\n✓ Building 6 composite scores (0-100 scale):")

# 4A) ACTIVITY SCORE: How engaged is the candidate on the platform right now?
df["activity_score"] = (
    df["profile_views_received_30d"].clip(0, 200) / 200 * 25 +
    df["search_appearance_30d"].clip(0, 500) / 500 * 20 +
    df["saved_by_recruiters_30d"].clip(0, 20) / 20 * 20 +
    df["applications_submitted_30d"].clip(0, 15) / 15 * 15 +
    (1 - df["days_since_active"].clip(0, 365) / 365) * 20
).round(2)
print(f"  1. Activity Score: {df['activity_score'].mean():.1f}±{df['activity_score'].std():.1f}")

# 4B) PROFILE QUALITY SCORE: Completeness, verifications, trust signals
df["profile_quality_score"] = (
    df["profile_completeness_score"] / 100 * 40 +
    df["verified_email"].astype(int) * 5 +
    df["verified_phone"].astype(int) * 5 +
    df["linkedin_connected"].astype(int) * 5 +
    df["connection_count"].clip(0, 500) / 500 * 15 +
    df["endorsements_received"].clip(0, 100) / 100 * 15 +
    df["github_active"] * 15
).round(2)
print(f"  2. Profile Quality: {df['profile_quality_score'].mean():.1f}±{df['profile_quality_score'].std():.1f}")

# 4C) CAREER PROGRESSION SCORE: Experience depth, stability, prestige
df["career_progression_score"] = (
    (df["num_jobs"].clip(1, 7) - 1) / 6 * 20 +  # Job variety
    df["avg_tenure_months"].clip(12, 48) / 48 * 30 +  # Tenure stability
    df["company_size_enc"] / 8 * 30 +  # Company prestige
    df["num_industries"].clip(1, 5) / 5 * 20  # Industry breadth
).round(2)
print(f"  3. Career Progression: {df['career_progression_score'].mean():.1f}±{df['career_progression_score'].std():.1f}")

# 4D) SKILL DEPTH SCORE: Technical capability and expertise
df["skill_depth_score"] = (
    (df["advanced_skills"] / df["num_skills"].replace(0, 1)) * 40 +  # % advanced
    df["total_endorsements"].clip(0, 300) / 300 * 30 +  # Peer validation
    df["avg_skill_duration_months"].clip(0, 60) / 60 * 15 +  # Skill longevity
    df["avg_assessment_score"] / 100 * 15  # Verified scores
).round(2)
print(f"  4. Skill Depth: {df['skill_depth_score'].mean():.1f}±{df['skill_depth_score'].std():.1f}")

# 4E) ENGAGEMENT SCORE: Responsiveness & commitment
df["engagement_score"] = (
    df["interview_completion_rate"] * 30 +
    df["offer_acceptance_rate"] * 20 +
    df["recruiter_response_rate"] * 30 +
    (1 - df["avg_response_time_hours"].clip(0, 200) / 200) * 20
).round(2)
print(f"  5. Engagement Score: {df['engagement_score'].mean():.1f}±{df['engagement_score'].std():.1f}")

# 4F) AVAILABILITY SCORE: Ease of hiring
df["availability_score"] = (
    df["open_to_work_flag"] * 40 +
    (1 - df["notice_period_days"].clip(0, 150) / 150) * 35 +
    df["willing_to_relocate"] * 25
).round(2)
print(f"  6. Availability Score: {df['availability_score'].mean():.1f}±{df['availability_score'].std():.1f}")

# 4G) INTERACTION FEATURES
print("\n✓ Building derived/interaction features:")

df["seniority_ratio"] = df["senior_roles_count"] / df["num_jobs"].replace(0, 1)
df["endorsement_per_skill"] = df["total_endorsements"] / df["num_skills"].replace(0, 1)
df["skill_versatility"] = df["num_skills"] * df["avg_skill_duration_months"] / 100
df["salary_range_width"] = df["salary_max"] - df["salary_min"]
df["response_efficiency"] = 1 / df["avg_response_time_hours"].clip(1, 500)

print(f"  - seniority_ratio, endorsement_per_skill, skill_versatility")
print(f"  - salary_range_width, response_efficiency")

# 4H) MASTER SCORE (Weighted composite of all 6 scores)
df["master_score"] = (
    df["career_progression_score"] * 0.25 +
    df["skill_depth_score"] * 0.25 +
    df["engagement_score"] * 0.20 +
    df["activity_score"] * 0.15 +
    df["profile_quality_score"] * 0.10 +
    df["availability_score"] * 0.05
).round(2)
print(f"\n✓ MASTER SCORE (target): {df['master_score'].mean():.1f}±{df['master_score'].std():.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: FEATURE SELECTION & CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 5: CORRELATION ANALYSIS")
print("="*80)

NUMERIC_FEATURES = [
    "years_of_experience", "num_jobs", "avg_tenure_months", "num_skills",
    "advanced_skills", "intermediate_skills", "beginner_skills",
    "total_endorsements", "avg_skill_duration_months",
    "num_certifications", "avg_assessment_score",
    "edu_tier_enc", "company_size_enc",
    "profile_completeness_score", "days_since_active", "days_on_platform",
    "github_activity_score", "connection_count", "endorsements_received",
    "salary_min", "salary_max", "salary_mid",
    "profile_views_received_30d", "search_appearance_30d",
    "saved_by_recruiters_30d", "applications_submitted_30d",
    "recruiter_response_rate", "interview_completion_rate",
    "offer_acceptance_rate", "avg_response_time_hours", "notice_period_days",
    "seniority_ratio", "endorsement_per_skill", "skill_versatility",
    "response_efficiency",
    "activity_score", "profile_quality_score", "career_progression_score",
    "skill_depth_score", "engagement_score", "availability_score",
    "master_score"
]

# Correlation with target (master_score)
print("\n✓ SPEARMAN CORRELATION WITH MASTER SCORE:")
correlations = {}
for feat in NUMERIC_FEATURES:
    if feat != "master_score":
        r, p = spearmanr(df[feat].fillna(0), df["master_score"])
        correlations[feat] = {"r": r, "p": p, "sig": "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""}

corr_df = pd.DataFrame(correlations).T.sort_values("r", ascending=False)
print("\nTop 15 POSITIVE correlations:")
for i, (feat, row) in enumerate(corr_df.head(15).iterrows(), 1):
    print(f"  {i:2d}. {feat:35s} r={row['r']:7.3f} p={row['p']:.3f} {row['sig']}")

print("\nTop 5 NEGATIVE correlations:")
for i, (feat, row) in enumerate(corr_df.tail(5).iterrows(), 1):
    print(f"  {i:2d}. {feat:35s} r={row['r']:7.3f} p={row['p']:.3f} {row['sig']}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 6: TRAIN / VAL / TEST SPLIT")
print("="*80)

# Select features for modeling
SELECTED_FEATURES = [
    "years_of_experience", "num_jobs", "avg_tenure_months", "num_skills",
    "advanced_skills", "total_endorsements", "avg_skill_duration_months",
    "num_certifications", "avg_assessment_score", "edu_tier_enc", "company_size_enc",
    "profile_completeness_score", "days_since_active", "github_activity_score",
    "connection_count", "salary_mid", "profile_views_received_30d",
    "applications_submitted_30d", "recruiter_response_rate",
    "interview_completion_rate", "offer_acceptance_rate",
    "seniority_ratio", "endorsement_per_skill", "skill_versatility",
    "open_to_work_flag", "willing_to_relocate", "work_mode_enc",
    "verified_email", "verified_phone", "linkedin_connected", "github_active"
]

TARGET = "master_score"

X = df[SELECTED_FEATURES].fillna(0)
y = df[TARGET]

# Normalize features to [0,1]
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=SELECTED_FEATURES)

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
)

print(f"\n✓ SPLIT RATIOS (70/15/15):")
print(f"  Train: {len(X_train):2d} candidates ({len(X_train)/50*100:5.1f}%)")
print(f"  Val:   {len(X_val):2d} candidates ({len(X_val)/50*100:5.1f}%)")
print(f"  Test:  {len(X_test):2d} candidates ({len(X_test)/50*100:5.1f}%)")

print(f"\n✓ FEATURE SCALING: MinMaxScaler [0,1]")
print(f"  {len(SELECTED_FEATURES)} features selected for modeling")

# Save split assignments back to df
split_assignment = pd.Series("", index=df.index)
split_assignment.iloc[X_train.index] = "train"
split_assignment.iloc[X_val.index] = "val"
split_assignment.iloc[X_test.index] = "test"
df["split"] = split_assignment

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 7: MODEL TRAINING & EVALUATION")
print("="*80)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3, learning_rate=0.1),
}

results = {}
for model_name, model in models.items():
    print(f"\n✓ Training {model_name}...")
    model.fit(X_train, y_train)
    
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test)
    ]:
        y_pred = model.predict(X_split)
        r2 = r2_score(y_split, y_pred)
        mae = mean_absolute_error(y_split, y_pred)
        rmse = np.sqrt(mean_squared_error(y_split, y_pred))
        
        results[f"{model_name}_{split_name}"] = {
            "r2": round(r2, 3),
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "predictions": y_pred,
            "actual": y_split.values,
            "model": model
        }
    
    print(f"  Train R²={results[f'{model_name}_train']['r2']}  Val R²={results[f'{model_name}_val']['r2']}  Test R²={results[f'{model_name}_test']['r2']}")

# Print full results table
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
print("\n{:<25} {:<15} {:<15} {:<15}".format("Model", "Train R²", "Val R²", "Test R²"))
print("-"*70)
for model_name in models.keys():
    train_r2 = results[f"{model_name}_train"]["r2"]
    val_r2 = results[f"{model_name}_val"]["r2"]
    test_r2 = results[f"{model_name}_test"]["r2"]
    print(f"{model_name:<25} {train_r2:<15.3f} {val_r2:<15.3f} {test_r2:<15.3f}")

print("\n{:<25} {:<15} {:<15} {:<15}".format("Model", "Train MAE", "Val MAE", "Test MAE"))
print("-"*70)
for model_name in models.keys():
    train_mae = results[f"{model_name}_train"]["mae"]
    val_mae = results[f"{model_name}_val"]["mae"]
    test_mae = results[f"{model_name}_test"]["mae"]
    print(f"{model_name:<25} {train_mae:<15.3f} {val_mae:<15.3f} {test_mae:<15.3f}")

# Get best model
best_model_name = max(
    ["Random Forest", "Gradient Boosting"],
    key=lambda m: results[f"{m}_test"]["r2"]
)
best_model = results[f"{best_model_name}_test"]["model"]
print(f"\n★ BEST MODEL: {best_model_name}")
print(f"  Test R² = {results[f'{best_model_name}_test']['r2']}")
print(f"  Test MAE = {results[f'{best_model_name}_test']['mae']}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 8: SAVING OUTPUTS")
print("="*80)

# Save complete dataset with features
df.to_csv(PROCESSED_DIR / "features_final.csv", index=False)
print(f"✓ Saved: features_final.csv ({df.shape[0]}x{df.shape[1]})")

# Save splits
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()
test_df = df[df["split"] == "test"].copy()

train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

print(f"✓ Saved: train.csv ({len(train_df)} records)")
print(f"✓ Saved: val.csv ({len(val_df)} records)")
print(f"✓ Saved: test.csv ({len(test_df)} records)")

# Save model predictions
predictions_df = pd.DataFrame({
    "candidate_id": df["candidate_id"],
    "split": df["split"],
    "actual_master_score": df["master_score"],
    "predicted_master_score": np.nan,
    "prediction_error": np.nan
})

for split, X_split, indices in [
    ("train", X_train, X_train.index),
    ("val", X_val, X_val.index),
    ("test", X_test, X_test.index)
]:
    preds = best_model.predict(X_split)
    predictions_df.loc[indices, "predicted_master_score"] = preds
    predictions_df.loc[indices, "prediction_error"] = preds - df.loc[indices, "master_score"].values

predictions_df.to_csv(PROCESSED_DIR / "model_predictions.csv", index=False)
print(f"✓ Saved: model_predictions.csv (all predictions + errors)")

# Save feature importance
if hasattr(best_model, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "feature": SELECTED_FEATURES,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    feature_importance.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)
    print(f"✓ Saved: feature_importance.csv (from {best_model_name})")

# Summary stats
summary = {
    "total_records": len(df),
    "total_features": len(SELECTED_FEATURES),
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "best_model": best_model_name,
    "best_test_r2": results[f"{best_model_name}_test"]["r2"],
    "best_test_mae": results[f"{best_model_name}_test"]["mae"],
    "timestamp": str(TODAY)
}
import json as json_module
with open(PROCESSED_DIR / "summary.json", "w") as f:
    json_module.dump(summary, f, indent=2)
print(f"✓ Saved: summary.json (metadata)")

print("\n" + "="*80)
print("✓ PIPELINE COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {PROCESSED_DIR}/")
print("\nGenerated files:")
print("  - features_final.csv        (all records with 80+ features)")
print("  - train.csv, val.csv, test.csv  (split data)")
print("  - model_predictions.csv     (predictions vs actual)")
print("  - feature_importance.csv    (feature ranking)")
print("  - summary.json              (pipeline metadata)")
print("\nNext: Generate 10 visualization plots (see notebooks/PLOTTING_NOTEBOOK.ipynb)")
