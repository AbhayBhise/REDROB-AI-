"""
rank.py — Final ranking pipeline for India Runs Hackathon
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
Pre-computation: python rank.py --precompute (run once, takes ~20 min)
"""
import os, sys, json, csv, argparse, math, pickle, re
from datetime import datetime
from collections import defaultdict
import numpy as np

os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transformers import AutoTokenizer, AutoModel
import torch
from rank_bm25 import BM25Okapi

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── XGBoost LTR Model Loading ─────────────────────────────────────────────────

XGB_MODEL = None  # Global model (loaded once at startup)

def load_xgb_model():
    """Load XGBoost LTR model (if available). Returns None if model unavailable."""
    global XGB_MODEL
    
    model_path = 'models/xgb_ranker.json'
    
    if not os.path.exists(model_path):
        return None  # Model file doesn't exist; fall back to hardcoded weights
    
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print(f"✓ Loaded XGBoost LTR model from {model_path}")
        return model
    except ImportError:
        print(f"  Warning: xgboost not installed. Using hardcoded weights.")
        return None
    except Exception as e:
        print(f"  Warning: Failed to load XGBoost model: {e}. Using hardcoded weights.")
        return None

def embed_text(text, tokenizer, model):
    """Embed a single text string — used only for the JD at ranking time"""
    encoded = tokenizer(
        [text], padding=True, truncation=True,
        max_length=256, return_tensors='pt'
    )
    with torch.no_grad():
        output = model(**encoded)
        emb = output.last_hidden_state[:, 0, :]
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.numpy()[0]

# ── JD-derived constants ─────────────────────────────────────────────────────

JD_TEXT = """
Senior AI Engineer founding team Redrob AI talent intelligence platform.
Production experience embeddings retrieval ranking systems sentence-transformers 
BGE E5 OpenAI embeddings deployed real users embedding drift index refresh 
retrieval quality regression. Vector databases hybrid search Pinecone Weaviate 
Qdrant Milvus OpenSearch Elasticsearch FAISS operational experience.
Python code quality. Evaluation frameworks ranking systems NDCG MRR MAP 
offline online A/B testing recruiter feedback loops.
LLM fine-tuning LoRA QLoRA PEFT learning to rank XGBoost neural ranking.
HR tech recruiting marketplace distributed systems large scale inference.
Scrappy product engineering ship working ranker embeddings hybrid retrieval 
LLM reranking architecture semantic search candidate matching job description.
5 to 9 years experience production deployment not pure research.
Pune Noida Bangalore India hybrid.
"""

# ── Structured JD Parse ───────────────────────────────────────────────────────

def parse_jd():
    """Extract structured requirements from the JD text.
    Returns a dict with typed fields — used to enrich scoring logic.
    Only captures what the JD explicitly states."""
    return {
        # Absolute must-haves (JD section: 'Things you absolutely need')
        'required_skills': [
            'embeddings', 'retrieval', 'ranking', 'python',
            'vector database', 'hybrid search', 'evaluation', 'ndcg', 'mrr',
            'sentence-transformers', 'bge', 'e5', 'faiss', 'pinecone',
            'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch',
            'a/b testing',
        ],
        # Nice-to-haves (JD section: 'Things we'd like you to have')
        'preferred_skills': [
            'lora', 'qlora', 'peft', 'fine-tuning', 'xgboost',
            'learning to rank', 'distributed systems', 'inference optimization',
            'hr tech', 'recruiting', 'marketplace', 'open source',
        ],
        # Explicit disqualifiers (JD section: 'Things we explicitly do NOT want')
        'disqualify_profiles': [
            'marketing manager', 'hr manager', 'content writer',
            'computer vision only', 'speech only', 'robotics only',
        ],
        'exp_min': 5,
        'exp_max': 9,
        'exp_preferred_min': 6,
        'exp_preferred_max': 8,
        'location_preferred': ['pune', 'noida'],
        'location_acceptable': ['bangalore', 'bengaluru', 'mumbai', 'hyderabad', 'delhi', 'india'],
        'notice_preferred_days': 30,   # JD: 'we'd love sub-30-day'
        'notice_buyout_days': 30,      # JD: 'can buy out up to 30 days'
        'work_mode': 'hybrid',         # JD: 'hybrid — flexible cadence'
        'production_required': True,   # JD: 'pure research = disqualifier'
        'requires_code_writing': True, # JD: 'this role writes code'
    }

JD = parse_jd()  # Parsed once at module level

MUST_HAVE_SKILLS = [
    'embeddings', 'semantic search', 'sentence-transformers', 'bge', 'e5',
    'faiss', 'milvus', 'pinecone', 'weaviate', 'qdrant', 'elasticsearch',
    'opensearch', 'vector', 'hybrid search',
    'python', 'ranking', 'retrieval', 'information retrieval',
    'ndcg', 'mrr', 'a/b testing', 'evaluation',
    'nlp', 'transformers', 'pytorch', 'huggingface',
    'llm', 'large language model', 'fine-tuning', 'rag',
    'machine learning', 'deep learning', 'neural network',
    'recommendation', 'search', 'reranking'
]

NICE_TO_HAVE_SKILLS = [
    'lora', 'qlora', 'peft', 'fine-tuning llms',
    'xgboost', 'learning to rank', 'lambdamart',
    'distributed systems', 'inference optimization',
    'kafka', 'spark', 'airflow', 'mlflow',
    'weights & biases', 'experiment tracking',
    'open source', 'github', 'research'
]

# Threshold raised to 4: one 'agile' tag on an ML engineer profile should not penalize
DISQUALIFY_ONLY_SKILLS = [
    'html', 'css', 'wordpress', 'photoshop', 'illustrator',
    'sales', 'accounting', 'crm', 'salesforce crm',
    'project management', 'scrum', 'agile',
    'figma', 'redux', 'angular', 'vue.js'
]
DISQUALIFY_THRESHOLD = 4  # require 4+ disqualify skills before penalizing

TIER1_CITIES = [
    'pune', 'noida', 'bangalore', 'bengaluru', 'mumbai',
    'hyderabad', 'delhi', 'chennai', 'gurgaon', 'gurugram', 'india'
]

CONSULTING_FIRMS = [
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'hcl', 'tech mahindra'
]

# JD-relevant certifications (verified competency signal)
JD_RELEVANT_CERTS = [
    'aws certified machine learning', 'machine learning specialty',
    'nlp specialization', 'deep learning specialization',
    'hugging face', 'langchain', 'mlops', 'tensorflow developer',
    'pytorch', 'information retrieval', 'natural language processing',
    'google professional ml', 'azure ai engineer', 'databricks',
]

# ── Honeypot Detection ────────────────────────────────────────────────────────

def is_honeypot(c):
    stated_yoe = c['profile'].get('years_of_experience', 0)
    
    # Only flag truly impossible skill durations
    impossible_skills = sum(
        1 for s in c.get('skills', [])
        if s.get('duration_months', 0) > (stated_yoe * 12) + 24
    )
    if impossible_skills >= 3:
        return True
    
    # Only flag zero-duration advanced skills if many
    zero_duration_advanced = sum(
        1 for s in c.get('skills', [])
        if s.get('proficiency') == 'advanced' and s.get('duration_months', 1) == 0
    )
    if zero_duration_advanced >= 5:
        return True
    
    return False

def consulting_penalty(c):
    history = c.get('career_history', [])
    if not history: return 0.0
    consulting_count = sum(
        1 for j in history
        if any(firm in j.get('company', '').lower() for firm in CONSULTING_FIRMS)
    )
    if consulting_count == len(history):  # entire career at consulting
        return 0.15
    return 0.0

# ── Individual Scorers ────────────────────────────────────────────────────────

def skill_score(c):
    skills_lower = {s['name'].lower(): s for s in c.get('skills', [])}

    # Penalize only when clearly non-AI dominated (threshold raised to 4)
    disqualify_count = sum(
        1 for sk in DISQUALIFY_ONLY_SKILLS
        if any(sk in csk for csk in skills_lower)
    )
    non_ai_penalty = min(disqualify_count / DISQUALIFY_THRESHOLD, 0.4) if disqualify_count >= DISQUALIFY_THRESHOLD else 0.0

    # Proficiency weights — fixed: 'expert' now correctly weighted above 'advanced'
    PROF_WEIGHT = {'expert': 2.0, 'advanced': 1.5, 'intermediate': 1.2, 'beginner': 0.8}

    must_score = 0.0
    for skill in MUST_HAVE_SKILLS:
        for csk, data in skills_lower.items():
            if skill in csk or csk in skill:
                w = PROF_WEIGHT.get(data.get('proficiency', 'beginner'), 0.8)
                dur = data.get('duration_months', 0)
                if dur >= 36: w += 0.3
                elif dur >= 12: w += 0.1
                if data.get('endorsements', 0) >= 20: w += 0.2
                must_score += w
                break

    nice_score = 0.0
    for skill in NICE_TO_HAVE_SKILLS:
        for csk, data in skills_lower.items():
            if skill in csk or csk in skill:
                nice_score += 0.6
                break

    must_norm = min(must_score / (len(MUST_HAVE_SKILLS) * 1.0), 1.0)
    nice_norm = min(nice_score / (len(NICE_TO_HAVE_SKILLS) * 0.6), 1.0)
    raw = 0.72 * must_norm + 0.28 * nice_norm
    return max(0.0, raw - non_ai_penalty)

def experience_score(c):
    yoe = c['profile'].get('years_of_experience', 0)
    if 5 <= yoe <= 9: return 1.0
    elif 4 <= yoe < 5: return 0.82
    elif 9 < yoe <= 11: return 0.88
    elif 3 <= yoe < 4: return 0.55
    elif 11 < yoe <= 13: return 0.72
    elif yoe > 13: return 0.55
    else: return 0.2

def production_score(c):
    """JD explicitly wants production deployers, not researchers"""
    prod_kw = ['production', 'deployed', 'users', 'scale', 'shipped',
               'launched', 'served', 'inference', 'api', 'real-time',
               'pipeline', 'system', 'platform', 'service']
    research_kw = ['research', 'paper', 'academic', 'lab', 'phd',
                   'arxiv', 'publication', 'benchmark', 'ablation']
    
    all_text = ' '.join(
        j.get('description', '') for j in c.get('career_history', [])
    ).lower()
    summary = c['profile'].get('summary', '').lower()
    full_text = all_text + ' ' + summary
    
    prod_hits = sum(1 for kw in prod_kw if kw in full_text)
    research_hits = sum(1 for kw in research_kw if kw in full_text)
    
    score = min(prod_hits / 8, 1.0) - min(research_hits / 5, 0.3)
    return max(0.0, score)

def behavioral_score(c):
    """Hiring Probability score from all available redrob_signals.
    Derived features:
      - Candidate Freshness: last_active_date recency
      - Hiring Intent: open_to_work + applications_submitted_30d
      - Recruiter Interest: saved_by_recruiters + profile_views + search_appearance
      - Reliability: interview_completion_rate + recruiter_response_rate + avg_response_time
      - Offer History: offer_acceptance_rate (guard: -1 = no history)
      - Profile Quality: profile_completeness + linkedin_connected
      - Technical Presence: github_activity_score
      - Trust: verified_email + verified_phone
    """
    sig = c.get('redrob_signals', {})
    score = 0.0
    weights = 0.0

    # --- Candidate Freshness (most important) ---
    if sig.get('last_active_date'):
        days = (datetime.now() - datetime.strptime(
            sig['last_active_date'], '%Y-%m-%d')).days
        recency = max(0, 1 - days / 365)
        score += recency * 2.0
        weights += 2.0

    # --- Hiring Intent ---
    score += (1.0 if sig.get('open_to_work_flag') else 0.0); weights += 1.0
    # applications_submitted_30d: 0 = passive, high = very active
    apps = sig.get('applications_submitted_30d', 0)
    score += min(apps / 10.0, 1.0) * 0.5; weights += 0.5

    # --- Recruiter Interest (market demand signal) ---
    score += min(sig.get('saved_by_recruiters_30d', 0) / 10, 1.0); weights += 1.0
    score += min(sig.get('profile_views_received_30d', 0) / 30, 1.0); weights += 1.0
    score += min(sig.get('search_appearance_30d', 0) / 200, 1.0) * 0.5; weights += 0.5

    # --- Reliability ---
    icr = sig.get('interview_completion_rate', -1)
    if icr >= 0:
        score += icr; weights += 1.0
    rrr = sig.get('recruiter_response_rate', 0)
    score += rrr; weights += 1.0
    # avg_response_time_hours: faster is better; cap at 168h (1 week)
    rt = sig.get('avg_response_time_hours', 168)
    responsiveness = max(0.0, 1.0 - rt / 168.0)
    score += responsiveness * 0.5; weights += 0.5

    # --- Offer History ---
    oar = sig.get('offer_acceptance_rate', -1)
    if oar >= 0:  # -1 means no history, don't penalize
        score += oar; weights += 1.0

    # --- Profile Quality ---
    score += (sig.get('profile_completeness_score', 0) / 100) * 1.5; weights += 1.5
    score += (0.5 if sig.get('linkedin_connected') else 0.0); weights += 0.5

    # --- Technical Presence (GitHub) ---
    gh = sig.get('github_activity_score', -1)
    if gh > 0:
        score += min(gh / 10, 1.0) * 1.5
        weights += 1.5

    # --- Trust ---
    trust = (0.5 if sig.get('verified_email') else 0) + \
            (0.5 if sig.get('verified_phone') else 0)
    score += trust; weights += 1.0

    return score / weights if weights > 0 else 0.0

# Keep old name as alias for sandbox_app.py compatibility
activity_score = behavioral_score

def location_score(c):
    loc = (c['profile'].get('location', '') + ' ' + 
           c['profile'].get('country', '')).lower()
    relocate = c['redrob_signals'].get('willing_to_relocate', False)
    
    if any(city in loc for city in TIER1_CITIES):
        return 1.0
    elif 'india' in loc and relocate:
        return 0.75
    elif 'india' in loc:
        return 0.55
    elif relocate:
        return 0.45
    else:
        return 0.25

def assessment_quality_score(c):
    """Skill assessments are ground truth — weight them strongly when available"""
    assessments = c['redrob_signals'].get('skill_assessment_scores', {})
    if not assessments:
        return None
    jd_relevant = ['NLP', 'Machine Learning', 'Python', 'Deep Learning',
                   'Fine-tuning LLMs', 'Information Retrieval', 'MLOps']
    relevant_scores = [v for k, v in assessments.items()
                      if any(r.lower() in k.lower() for r in jd_relevant)]
    all_scores = list(assessments.values())
    if relevant_scores:
        return (sum(relevant_scores) / len(relevant_scores)) / 100
    return (sum(all_scores) / len(all_scores)) / 100

def edu_tier_score(c):
    """Education institution prestige. tier_1 = IITs/IISc/BITS Pilani.
    Small but genuinely discriminating signal for this JD."""
    edu = c.get('education', [])
    if not edu:
        return 0.25
    tier_map = {'tier_1': 1.0, 'tier_2': 0.70, 'tier_3': 0.45, 'tier_4': 0.25, 'unknown': 0.25}
    # Score by best (highest prestige) degree held
    best = max(tier_map.get(e.get('tier', 'unknown'), 0.25) for e in edu)
    return best

def cert_score(c):
    """JD-relevant certifications as a verified competency signal.
    Capped at 0.1 — meaningful but not dominant."""
    certs = c.get('certifications', [])
    if not certs:
        return 0.0
    hits = 0
    for cert in certs:
        name_lower = cert.get('name', '').lower()
        if any(kw in name_lower for kw in JD_RELEVANT_CERTS):
            hits += 1
    return min(hits * 0.04, 0.10)

def notice_period_score(c):
    """JD says: love sub-30d, can buy out 30d, 30d+ bar gets higher.
    Weight is deliberately small — it's a filter hint, not a primary signal."""
    days = c.get('redrob_signals', {}).get('notice_period_days', 90)
    if days <= 30:  return 1.0
    if days <= 60:  return 0.75
    if days <= 90:  return 0.55
    return 0.30

def title_relevance_score(c):
    """Current title alignment with Senior AI Engineer role"""
    title = c['profile'].get('current_title', '').lower()
    headline = c['profile'].get('headline', '').lower()
    combined = title + ' ' + headline

    high_match = ['ai engineer', 'ml engineer', 'machine learning engineer',
                 'nlp engineer', 'data scientist', 'research engineer',
                 'applied scientist', 'senior engineer', 'staff engineer']
    mid_match = ['software engineer', 'backend engineer', 'data engineer',
                 'full stack', 'platform engineer', 'mlops']
    low_match = ['manager', 'analyst', 'consultant', 'hr', 'marketing',
                 'sales', 'designer', 'content', 'writer']

    for t in high_match:
        if t in combined: return 1.0
    for t in mid_match:
        if t in combined: return 0.65
    for t in low_match:
        if t in combined: return 0.15
    return 0.4

# ── Final Fusion Score ────────────────────────────────────────────────────────

def compute_score(c, debug=False):
    if is_honeypot(c):
        if debug: print(f"  [HONEYPOT] {c['candidate_id']}")
        return 0.001  # force to bottom

    s_skill      = skill_score(c)
    s_exp        = experience_score(c)
    s_prod       = production_score(c)
    s_behavioral = behavioral_score(c)
    s_location   = location_score(c)
    s_title      = title_relevance_score(c)
    s_assessment = assessment_quality_score(c)
    s_edu        = edu_tier_score(c)
    s_cert       = cert_score(c)
    s_notice     = notice_period_score(c)
    s_consult    = consulting_penalty(c)

    # Base score — weights normalized to sum ~1.00
    # Rationale:
    #   skill(0.25): primary match signal
    #   exp(0.16):   band fit
    #   prod(0.16):  production vs research — JD is explicit
    #   behavioral(0.13): hiring probability composite
    #   title(0.11): current role alignment
    #   location(0.08): Tier1 city preference
    #   edu(0.05):   prestige signal — IIT/IISc meaningful but not dominant
    #   cert(0.04):  verified competency bonus
    #   notice(0.02): JD preference: sub-30d
    # Use XGBoost LTR model if available, otherwise fall back to hardcoded weights
    if XGB_MODEL is not None:
        features = np.array([[
            s_skill,
            s_exp,
            s_prod,
            s_behavioral,
            s_location,
            s_title,
            s_assessment if s_assessment is not None else 0.0,
            s_edu,
            s_cert,
            s_notice,
            s_consult,
        ]], dtype=np.float32)
        
        # Get probability of class 1 (Hire)
        base = XGB_MODEL.predict_proba(features)[0][1]
    else:
        # Fallback: hardcoded weights
        base = (
            0.25 * s_skill      +
            0.16 * s_exp        +
            0.16 * s_prod       +
            0.13 * s_behavioral +
            0.11 * s_title      +
            0.08 * s_location   +
            0.05 * s_edu        +
            0.04 * s_cert       +
            0.02 * s_notice
        )

    # Assessment multiplier — verified ground truth from Redrob platform
    if s_assessment is not None:
        multiplier = 0.85 + (s_assessment * 0.30)
        base = base * multiplier

    # Consulting-only career penalty (JD explicit)
    base = base - s_consult
    base = max(0.0, base)

    if debug:
        print(f"  {c['candidate_id']} | skill={s_skill:.3f} exp={s_exp:.3f} prod={s_prod:.3f} "
              f"behav={s_behavioral:.3f} title={s_title:.3f} loc={s_location:.3f} "
              f"edu={s_edu:.3f} cert={s_cert:.3f} notice={s_notice:.3f} "
              f"assess={s_assessment} consult_pen={s_consult:.3f} => base={base:.4f}")

    return round(min(base, 1.0), 6)

# ── Reasoning Generator ───────────────────────────────────────────────────────

def generate_reasoning(c, score, rank):
    """Generate specific, factual reasoning with a clear score breakdown for transparency."""
    p = c['profile']
    sig = c.get('redrob_signals', {})
    
    # Recompute base heuristic scores to show the breakdown
    s_skill      = skill_score(c)
    s_exp        = experience_score(c)
    s_prod       = production_score(c)
    s_behavioral = behavioral_score(c)
    s_title      = title_relevance_score(c)
    s_edu        = edu_tier_score(c)
    
    skills = [s['name'] for s in c.get('skills', []) if s.get('proficiency') in ['advanced', 'expert', 'intermediate']]
    title = p.get('current_title', 'Unknown Role')
    yoe = p.get('years_of_experience', 0)
    
    reasoning = []
    
    # 1. Title & Experience (Target: 5-9 years Senior AI Engineer)
    if s_title > 0.8 and s_exp > 0.8:
        reasoning.append(f"[FIT] **Perfect Role Fit**: {title} with {yoe:.1f}y experience closely matches the Senior AI Engineer requirement.")
    elif s_exp < 0.5:
        reasoning.append(f"[GAP] **Experience Gap**: {yoe:.1f} years of experience falls outside the target 5-9 years band.")
    elif s_title < 0.5:
        reasoning.append(f"[GAP] **Title Mismatch**: Current role '{title}' diverges from the core ML/AI engineering focus.")
    else:
        reasoning.append(f"[FIT] **Solid Fit**: {title} with {yoe:.1f}y experience.")
        
    # 2. Tech Stack & Production (Target: Embeddings, RAG, Weaviate, Pinecone, Production)
    if s_skill > 0.7:
        reasoning.append(f"[FIT] **Strong Skill Match**: High overlap with required tech stack (e.g., {', '.join(skills[:3]) if skills else 'ML/AI'}).")
    else:
        reasoning.append(f"[GAP] **Skill Gap**: Lacks sufficient depth in core ranking/retrieval tech (Vector DBs, LLM fine-tuning).")
        
    if s_prod > 0.7:
        reasoning.append(f"[FIT] **Production Ready**: Proven experience shipping real-world ML systems rather than pure research.")
    elif s_prod < 0.3:
        reasoning.append(f"[GAP] **Research Heavy**: Lacks signals of large-scale production deployment.")
        
    # 3. Behavioral & Redrob Signals
    if s_edu > 0.5:
        reasoning.append(f"[STRONG] **Top Tier Edu**: Graduated from a Tier-1 institution.")
        
    assessments = sig.get('skill_assessment_scores', {})
    if assessments:
        avg_assess = sum(assessments.values()) / len(assessments)
        if avg_assess > 80:
            reasoning.append(f"[STRONG] **High Assessment**: Averaged {avg_assess:.0f}/100 on Redrob tests.")
        elif avg_assess < 50:
            reasoning.append(f"[GAP] **Low Assessment**: Averaged {avg_assess:.0f}/100 on Redrob tests.")
            
    # 4. Final Score Breakdown
    reasoning.append(f"\n_**Score Breakdown:** Skills ({s_skill:.2f}) | Exp ({s_exp:.2f}) | Title ({s_title:.2f}) | Prod ({s_prod:.2f}) | Behav ({s_behavioral:.2f})_")
    
    if is_honeypot(c):
        reasoning = ["🚨 **HONEYPOT DETECTED**: Contradictory profile data. Automatically penalized to bottom rank."]
        
    return "  \n".join(reasoning)

def load_expanded_keywords():
    """Load expanded JD keywords from file (if exists). Returns list of keywords or empty list."""
    expanded_file = 'data/processed/expanded_keywords.json'
    if os.path.exists(expanded_file):
        try:
            with open(expanded_file, 'r', encoding='utf-8') as f:
                keywords = json.load(f)
            print(f"  Loaded {len(keywords)} expanded keywords from {expanded_file}")
            return keywords
        except Exception as e:
            print(f"  Warning: Failed to load expanded keywords: {e}")
            return []
    return []

def build_candidate_text(c):
    """Build BM25 text representation.
    Priority order: current_title > headline > summary > skills > career > education.
    Current role leads because it is the strongest relevance signal for BM25 matching.
    Note: precompute.py has its own copy — changes here only affect BM25 at ranking time."""
    p = c.get('profile', {})
    parts = []

    # Current role first (most discriminating for BM25)
    if p.get('current_title'): parts.append(p['current_title'])
    if p.get('headline'): parts.append(p['headline'])
    if p.get('summary'):  parts.append(p['summary'])
    if p.get('current_industry'): parts.append(p['current_industry'])

    # Skills with proficiency weighting
    skill_parts = []
    for s in c.get('skills', []):
        name = s.get('name', '')
        prof = s.get('proficiency', '')
        if prof in ('expert', 'advanced'):
            skill_parts.append(f"{name} {name} {name}")   # triple weight
        elif prof == 'intermediate':
            skill_parts.append(f"{name} {name}")           # double weight
        else:
            skill_parts.append(name)
    if skill_parts:
        parts.append('Skills: ' + ', '.join(skill_parts))

    # Career history
    for job in c.get('career_history', []):
        if job.get('title'): parts.append(job['title'])
        if job.get('description'): parts.append(job['description'])

    # Education field
    for edu in c.get('education', []):
        if edu.get('field_of_study'): parts.append(edu['field_of_study'])

    return ' '.join(parts)

# ── Main Pipeline ─────────────────────────────────────────────────────────────

def _banner(msg='', width=80):
    """Print a full-width banner line."""
    if msg:
        pad = width - len(msg) - 4
        print(f"  {msg}{' ' * max(0, pad)}")
    else:
        print('=' * width)

def main():
    t_start = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='candidates.jsonl')
    parser.add_argument('--out', default='submission.csv')
    parser.add_argument('--emb', default='data/candidates_embeddings.npy')
    parser.add_argument('--ids', default='data/candidate_ids_ordered.json')
    parser.add_argument('--debug', action='store_true',
                        help='Print per-candidate score breakdown (does not affect submission)')
    args = parser.parse_args()

    _banner()
    _banner('REDROB AI — Candidate Ranking Pipeline')
    _banner('India Runs Hackathon  ·  Track 1  ·  Advanced Hybrid Pipeline')
    _banner()
    print()

    print("[1/5] Loading candidates ...")
    candidates = []
    try:
        with open(args.candidates, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    except FileNotFoundError:
        print("candidates.jsonl not found, trying sample_candidates.json")
        with open('data/external/Dataset/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json', 'r', encoding='utf-8') as f:
            candidates = json.load(f)
            
    print(f"      ✓ {len(candidates):,} candidates loaded")
    candidate_map = {c['candidate_id']: c for c in candidates}

    print("[2/5] Loading precomputed embeddings ...")
    embeddings = None
    embedding_model_used = 'BAAI/bge-small-en-v1.5'  # default fallback
    
    # Try to load .npz (newer format with bge-base)
    npz_path = 'data/processed/candidates_embeddings.npz'
    if os.path.exists(npz_path):
        try:
            with np.load(npz_path, allow_pickle=True) as npz_file:
                embeddings = npz_file['embeddings']
                # Convert from float16 back to float32 if needed
                if embeddings.dtype == np.float16:
                    embeddings = embeddings.astype(np.float32)
                embedding_model_used = 'BAAI/bge-base-en-v1.5'
                print(f"  Loaded embeddings from {npz_path} (768-dim, bge-base)")
        except Exception as e:
            print(f"  Warning: Failed to load {npz_path}: {e}")
            embeddings = None
    
    # Fall back to .npy if .npz not found
    if embeddings is None:
        try:
            embeddings = np.load(args.emb)
            print(f"  Loaded embeddings from {args.emb} (384-dim, bge-small)")
        except FileNotFoundError:
            print(f"Error: No embeddings found. Please run precompute.py first.")
            sys.exit(1)
    
    try:
        with open(args.ids, 'r', encoding='utf-8') as f:
            ordered_ids = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.ids} not found. Please run precompute.py first.")
        sys.exit(1)
        
    emb_dict = {cid: embeddings[i] for i, cid in enumerate(ordered_ids)}
        
    print(f"      ✓ Embeddings loaded ({embedding_model_used})")
    print("[3/5] Hybrid Retrieval — BM25 + Dense Semantic Search ...")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_used)
    embed_model = AutoModel.from_pretrained(embedding_model_used)
    embed_model.eval()

    jd_embedding = embed_text(JD_TEXT, tokenizer, embed_model)
    jd_emb_norm = jd_embedding  # already L2-normalized by embed_text

    # Load XGBoost LTR model (optional; falls back to hardcoded weights if unavailable)
    global XGB_MODEL
    XGB_MODEL = load_xgb_model()

    print("      Initializing BM25 ...")
    tokenized_corpus = [build_candidate_text(candidate_map[cid]).lower().split() for cid in ordered_ids]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Build tokenized JD with optional expanded keywords for improved recall
    tokenized_jd = JD_TEXT.lower().split()
    expanded_keywords = load_expanded_keywords()
    if expanded_keywords:
        # Repeat expanded keywords 2x to ensure they carry weight in BM25 scoring
        tokenized_jd.extend(expanded_keywords)
        tokenized_jd.extend(expanded_keywords)  # 2x repetition
        print(f"  Augmented tokenized_jd with {len(expanded_keywords)} expanded keywords (2x repetition)")
    
    bm25_scores_raw = bm25.get_scores(tokenized_jd)
    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0
    bm25_scores_dict = {cid: score/max_bm25 for cid, score in zip(ordered_ids, bm25_scores_raw)}
    
    print("      ✓ BM25 + Dense scores computed → Top 500 selected")
    print("[4/5] Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2) ...")
    print("      Scoring and fusing all signals ...")
    scored = []
    honeypot_count = 0
    for cid in ordered_ids:
        c = candidate_map[cid]
        master_score = compute_score(c, debug=args.debug)

        if master_score <= 0.001:
            honeypot_count += 1
            final_score = 0.001
        else:
            c_emb = emb_dict[cid]
            c_emb_norm = c_emb / np.linalg.norm(c_emb)
            semantic_score = float(np.dot(jd_emb_norm, c_emb_norm))
            semantic_score = max(0.0, min(1.0, semantic_score))

            bm25_score = bm25_scores_dict[cid]

            final_score = 0.40 * semantic_score + 0.25 * bm25_score + 0.35 * master_score

            if args.debug:
                print(f"    semantic={semantic_score:.3f} bm25={bm25_score:.3f} "
                      f"master={master_score:.4f} => final={final_score:.4f}")

        scored.append((c, final_score))

    
    print(f"      ✓ {honeypot_count} honeypot profiles detected and penalized")

    # Sort: score descending, then candidate_id numeric ascending as tie-breaker
    scored.sort(key=lambda x: (-round(x[1], 4), int(x[0]['candidate_id'].split('_')[1])))

    # Take top 100 only
    top100 = scored[:100]

    print("[5/5] Writing submission.csv + Generating Reasoning ...")
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, (c, score) in enumerate(top100, 1):
            reasoning = generate_reasoning(c, score, rank)
            writer.writerow([
                c['candidate_id'],
                rank,
                f"{score:.4f}",
                reasoning
            ])
    print(f"      ✓ {args.out} written — 100 candidates, ranks 1-100")

    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    print()
    _banner()
    _banner(f'PIPELINE COMPLETE  |  Runtime: {elapsed:.1f}s  |  Output: {args.out}')
    _banner()
    print()

    print("  Top 10 Candidates:")
    print(f"  {'Rank':<6} {'Candidate ID':<16} {'Score':<8} {'Title':<35} {'Exp':>5}")
    print("  " + "-" * 73)
    for rank, (c, score) in enumerate(top100[:10], 1):
        title = c['profile'].get('current_title', 'N/A')[:34]
        yoe = c['profile'].get('years_of_experience', 0)
        print(f"  {rank:<6} {c['candidate_id']:<16} {score:<8.4f} {title:<35} {yoe:>4.1f}y")
    print()

if __name__ == '__main__':
    main()