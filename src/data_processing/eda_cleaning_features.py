"""
India Runs Hackathon — Track 1: Data & AI Challenge
EDA + Cleaning + Feature Extraction
Team use: run this on sample_candidates.json (or stream from candidates.jsonl)
Output: features_extracted.csv (one row per candidate, 31 features, 0 nulls)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("./Dataset/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/")           # folder containing the dataset files
TODAY      = datetime(2026, 6, 22)

# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────
# For sample_candidates.json (50 records, used here for EDA):
with open(DATA_DIR / "sample_candidates.json") as f:
    raw = json.load(f)

# For the full candidates.jsonl (stream line-by-line to avoid memory issues):
# raw = []
# with open(DATA_DIR / "candidates.jsonl") as f:
#     for line in f:
#         raw.append(json.loads(line))

print(f"Loaded {len(raw)} candidate records")

# ── STEP 2: FLATTEN (one dict per candidate) ──────────────────────────────────
records = []
for c in raw:
    rec = {"candidate_id": c["candidate_id"]}

    # ── profile ──
    rec.update(c["profile"])

    # ── career_history ──
    ch = c["career_history"]
    rec["num_jobs"]            = len(ch)
    rec["total_career_months"] = sum(x.get("duration_months", 0) or 0 for x in ch)
    current                    = [x for x in ch if x.get("is_current")]
    rec["current_tenure_months"] = current[0]["duration_months"] if current else 0
    rec["has_current_job"]     = len(current) > 0
    rec["num_industries"]      = len(set(x.get("industry", "") for x in ch))
    rec["avg_tenure_months"]   = round(
        rec["total_career_months"] / rec["num_jobs"], 1
    ) if rec["num_jobs"] else 0

    # ── education ──
    edu  = c["education"]
    rec["num_degrees"]      = len(edu)
    tiers = [e.get("tier", "") for e in edu]
    rec["highest_edu_tier"] = tiers[0] if tiers else None
    rec["has_tier1_edu"]    = any("tier_1" in t for t in tiers)

    # ── skills ──
    sk = c["skills"]
    rec["num_skills"]               = len(sk)
    rec["advanced_skills"]          = sum(1 for s in sk if s.get("proficiency") == "advanced")
    rec["intermediate_skills"]      = sum(1 for s in sk if s.get("proficiency") == "intermediate")
    rec["beginner_skills"]          = sum(1 for s in sk if s.get("proficiency") == "beginner")
    rec["total_endorsements"]       = sum(s.get("endorsements", 0) for s in sk)
    rec["avg_skill_duration_months"] = round(
        np.mean([s.get("duration_months", 0) for s in sk]), 1
    ) if sk else 0

    # ── certifications & languages ──
    rec["num_certifications"] = len(c.get("certifications", []))
    langs = c.get("languages", [])
    rec["num_languages"]  = len(langs)
    rec["speaks_english"] = any(l["language"] == "English" for l in langs)

    # ── redrob_signals ──
    sig = c["redrob_signals"]
    for k, v in sig.items():
        if k not in ("expected_salary_range_inr_lpa", "skill_assessment_scores"):
            rec[k] = v

    salary = sig.get("expected_salary_range_inr_lpa", {})
    rec["salary_min"] = salary.get("min")
    rec["salary_max"] = salary.get("max")
    rec["salary_mid"] = round(
        (salary.get("min", 0) + salary.get("max", 0)) / 2, 1
    ) if salary else None

    ass = sig.get("skill_assessment_scores", {})
    rec["num_assessments"]      = len(ass)
    rec["avg_assessment_score"] = round(np.mean(list(ass.values())), 1) if ass else None

    records.append(rec)

df = pd.DataFrame(records)
print(f"Raw flattened shape: {df.shape}")

# ── STEP 3: DATA QUALITY REPORT ───────────────────────────────────────────────
print("\n=== NULL / ANOMALY CHECK ===")
print("Null counts:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\ngithub_activity_score = -1  → {(df['github_activity_score']==-1).sum()} records (no GitHub linked)")
print(f"offer_acceptance_rate = -1  → {(df['offer_acceptance_rate']==-1).sum()} records (no offers received)")
print(f"avg_assessment_score = null → {df['avg_assessment_score'].isna().sum()} records (no assessments taken)")

# ── STEP 4: CLEANING ──────────────────────────────────────────────────────────

# Parse dates → derive recency features
df["signup_date"]      = pd.to_datetime(df["signup_date"])
df["last_active_date"] = pd.to_datetime(df["last_active_date"])
df["days_since_active"] = (TODAY - df["last_active_date"]).dt.days
df["days_on_platform"]  = (TODAY - df["signup_date"]).dt.days

# Sentinel -1 values: clip to 0 (sentinel means "not available", not negative)
df["github_activity_score_clean"] = df["github_activity_score"].clip(lower=0)
df["offer_acceptance_rate_clean"] = df["offer_acceptance_rate"].clip(lower=0)

# Binary flag: did the candidate bother linking GitHub?
df["github_active"] = (df["github_activity_score"] > 0).astype(int)

# Missing assessment score: fill with 0 (no assessments taken = no score credit)
df["avg_assessment_score_clean"] = df["avg_assessment_score"].fillna(0)

# Ordinal encode education tier (higher = better institution)
edu_tier_map = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1}
df["edu_tier_score"] = df["highest_edu_tier"].map(edu_tier_map).fillna(1).astype(int)

# Ordinal encode company size (larger company = higher prestige signal)
size_order = {
    "1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4,
    "501-1000": 5, "1001-5000": 6, "5001-10000": 7, "10001+": 8
}
df["company_size_score"] = df["current_company_size"].map(size_order)

print("\n=== AFTER CLEANING ===")
print(f"Remaining nulls: {df.isnull().sum().sum()}")

# ── STEP 5: FEATURE ENGINEERING ───────────────────────────────────────────────
# Six composite scores, each 0–100, from weighted sub-signals.
# Weights are starting points — tune with ablation on Day 4.

# 1) ACTIVITY SCORE — how engaged is this candidate with the platform right now?
df["activity_score"] = (
    df["profile_views_received_30d"].clip(0, 200) / 200 * 25 +
    df["search_appearance_30d"].clip(0, 500) / 500 * 20 +
    df["saved_by_recruiters_30d"].clip(0, 20) / 20 * 20 +
    df["applications_submitted_30d"].clip(0, 15) / 15 * 15 +
    (1 - df["days_since_active"].clip(0, 365) / 365) * 20
).round(2)

# 2) PROFILE QUALITY SCORE — how complete and trustworthy is this profile?
df["profile_quality_score"] = (
    df["profile_completeness_score"] / 100 * 40 +
    df["verified_email"].astype(int) * 5 +
    df["verified_phone"].astype(int) * 5 +
    df["linkedin_connected"].astype(int) * 5 +
    df["connection_count"].clip(0, 500) / 500 * 15 +
    df["endorsements_received"].clip(0, 100) / 100 * 15 +
    df["github_active"] * 15
).round(2)

# 3) CAREER PROGRESSION SCORE — seniority, stability, breadth
df["career_progression_score"] = (
    (df["num_jobs"].clip(1, 7) - 1) / 6 * 20 +   # variety (more jobs = more breadth)
    df["avg_tenure_months"].clip(12, 48) / 48 * 30 +  # stability (24–48mo ideal)
    df["company_size_score"] / 8 * 30 +             # company prestige
    df["num_industries"].clip(1, 5) / 5 * 20         # industry breadth
).round(2)

# 4) SKILL DEPTH SCORE — quality over quantity of skills
df["skill_depth_score"] = (
    (df["advanced_skills"] / df["num_skills"].replace(0, 1)) * 40 +  # % advanced
    df["total_endorsements"].clip(0, 300) / 300 * 30 +                # peer validation
    df["avg_skill_duration_months"].clip(0, 60) / 60 * 15 +           # longevity
    df["avg_assessment_score_clean"] / 100 * 15                        # verified scores
).round(2)

# 5) ENGAGEMENT / RELIABILITY SCORE — will this person actually respond and show up?
df["engagement_score"] = (
    df["interview_completion_rate"] * 30 +
    df["offer_acceptance_rate_clean"] * 20 +
    df["recruiter_response_rate"] * 30 +
    (1 - df["avg_response_time_hours"].clip(0, 200) / 200) * 20
).round(2)

# 6) AVAILABILITY SCORE — how easily can they join?
df["availability_score"] = (
    df["open_to_work_flag"].astype(int) * 40 +
    (1 - df["notice_period_days"].clip(0, 150) / 150) * 35 +
    df["willing_to_relocate"].astype(int) * 25
).round(2)

# ── STEP 6: SELECT FINAL FEATURE SET ─────────────────────────────────────────
FEATURE_COLS = [
    # Identity
    "candidate_id",
    # Core profile
    "years_of_experience", "num_jobs", "avg_tenure_months", "num_industries",
    "current_tenure_months", "has_current_job",
    # Skills
    "num_skills", "advanced_skills", "intermediate_skills", "beginner_skills",
    "total_endorsements", "avg_skill_duration_months",
    # Education & certs
    "num_degrees", "edu_tier_score", "has_tier1_edu", "num_certifications",
    # Assessments
    "num_assessments", "avg_assessment_score_clean",
    # Company signal
    "company_size_score",
    # Platform engagement
    "profile_completeness_score", "days_since_active", "days_on_platform",
    "github_activity_score_clean", "github_active",
    "connection_count", "endorsements_received",
    # Compensation
    "salary_min", "salary_max", "salary_mid",
    # Availability
    "open_to_work_flag", "willing_to_relocate", "preferred_work_mode",
    "notice_period_days",
    # Languages
    "num_languages", "speaks_english",
    # Composite scores (use these directly in ranking)
    "activity_score", "profile_quality_score", "career_progression_score",
    "skill_depth_score", "engagement_score", "availability_score",
]

df_features = df[FEATURE_COLS].copy()

print("\n=== FINAL FEATURE SET ===")
print(f"Shape: {df_features.shape}")
print(f"Null values: {df_features.isnull().sum().sum()}")
print("\nComposite score summary:")
scores = ["activity_score","profile_quality_score","career_progression_score",
          "skill_depth_score","engagement_score","availability_score"]
print(df_features[scores].describe().round(2).to_string())

# ── STEP 7: SAVE ──────────────────────────────────────────────────────────────
df_features.to_csv("features_extracted.csv", index=False)
df.to_csv("candidates_full_cleaned.csv", index=False)
print("\nSaved: features_extracted.csv (use for ranking pipeline)")
print("Saved: candidates_full_cleaned.csv (full cleaned data for debugging)")


# ── STEP 8: HELPER — process a single candidate at inference time ──────────────
def extract_features_single(candidate: dict) -> dict:
    """
    Extract the same feature set from one raw candidate dict.
    Use this at inference time when ranking against a job description.
    """
    c = candidate
    rec = {"candidate_id": c["candidate_id"]}
    rec.update(c["profile"])
    ch = c["career_history"]
    rec["num_jobs"]              = len(ch)
    rec["total_career_months"]   = sum(x.get("duration_months",0) or 0 for x in ch)
    current                      = [x for x in ch if x.get("is_current")]
    rec["current_tenure_months"] = current[0]["duration_months"] if current else 0
    rec["has_current_job"]       = len(current) > 0
    rec["num_industries"]        = len(set(x.get("industry","") for x in ch))
    rec["avg_tenure_months"]     = rec["total_career_months"]/rec["num_jobs"] if rec["num_jobs"] else 0
    edu   = c["education"]
    tiers = [e.get("tier","") for e in edu]
    rec["num_degrees"]      = len(edu)
    rec["highest_edu_tier"] = tiers[0] if tiers else None
    rec["has_tier1_edu"]    = any("tier_1" in t for t in tiers)
    rec["edu_tier_score"]   = edu_tier_map.get(rec["highest_edu_tier"], 1)
    sk = c["skills"]
    rec["num_skills"]                = len(sk)
    rec["advanced_skills"]           = sum(1 for s in sk if s.get("proficiency")=="advanced")
    rec["intermediate_skills"]       = sum(1 for s in sk if s.get("proficiency")=="intermediate")
    rec["beginner_skills"]           = sum(1 for s in sk if s.get("proficiency")=="beginner")
    rec["total_endorsements"]        = sum(s.get("endorsements",0) for s in sk)
    rec["avg_skill_duration_months"] = np.mean([s.get("duration_months",0) for s in sk]) if sk else 0
    rec["num_certifications"] = len(c.get("certifications",[]))
    langs = c.get("languages",[])
    rec["num_languages"]  = len(langs)
    rec["speaks_english"] = any(l["language"]=="English" for l in langs)
    sig   = c["redrob_signals"]
    salary = sig.get("expected_salary_range_inr_lpa",{})
    rec["salary_min"] = salary.get("min")
    rec["salary_max"] = salary.get("max")
    rec["salary_mid"] = (salary.get("min",0)+salary.get("max",0))/2 if salary else None
    ass = sig.get("skill_assessment_scores",{})
    rec["num_assessments"]           = len(ass)
    rec["avg_assessment_score_clean"] = np.mean(list(ass.values())) if ass else 0
    rec["github_activity_score_clean"] = max(0, sig.get("github_activity_score",0))
    rec["github_active"]              = int(sig.get("github_activity_score",0) > 0)
    rec["offer_acceptance_rate_clean"] = max(0, sig.get("offer_acceptance_rate",0))
    rec["company_size_score"] = size_order.get(c["profile"].get("current_company_size",""),1)
    last_active = datetime.fromisoformat(sig["last_active_date"])
    signup      = datetime.fromisoformat(sig["signup_date"])
    rec["days_since_active"] = (TODAY - last_active).days
    rec["days_on_platform"]  = (TODAY - signup).days
    for k in ["profile_completeness_score","profile_views_received_30d","applications_submitted_30d",
              "recruiter_response_rate","avg_response_time_hours","connection_count","endorsements_received",
              "notice_period_days","preferred_work_mode","willing_to_relocate","open_to_work_flag",
              "search_appearance_30d","saved_by_recruiters_30d","interview_completion_rate",
              "verified_email","verified_phone","linkedin_connected"]:
        rec[k] = sig.get(k)
    # Composite scores
    rec["activity_score"] = (
        min(rec["profile_views_received_30d"],200)/200*25 +
        min(rec["search_appearance_30d"],500)/500*20 +
        min(rec["saved_by_recruiters_30d"],20)/20*20 +
        min(rec["applications_submitted_30d"],15)/15*15 +
        (1-min(rec["days_since_active"],365)/365)*20
    )
    rec["profile_quality_score"] = (
        rec["profile_completeness_score"]/100*40 +
        int(rec["verified_email"])*5 + int(rec["verified_phone"])*5 +
        int(rec["linkedin_connected"])*5 +
        min(rec["connection_count"],500)/500*15 +
        min(rec["endorsements_received"],100)/100*15 +
        rec["github_active"]*15
    )
    rec["career_progression_score"] = (
        (min(rec["num_jobs"],7)-1)/6*20 +
        min(max(rec["avg_tenure_months"],12),48)/48*30 +
        rec["company_size_score"]/8*30 +
        min(rec["num_industries"],5)/5*20
    )
    rec["skill_depth_score"] = (
        (rec["advanced_skills"]/max(rec["num_skills"],1))*40 +
        min(rec["total_endorsements"],300)/300*30 +
        min(rec["avg_skill_duration_months"],60)/60*15 +
        rec["avg_assessment_score_clean"]/100*15
    )
    rec["engagement_score"] = (
        rec["interview_completion_rate"]*30 +
        rec["offer_acceptance_rate_clean"]*20 +
        rec["recruiter_response_rate"]*30 +
        (1-min(rec["avg_response_time_hours"],200)/200)*20
    )
    rec["availability_score"] = (
        int(rec["open_to_work_flag"])*40 +
        (1-min(rec["notice_period_days"],150)/150)*35 +
        int(rec["willing_to_relocate"])*25
    )
    return rec
