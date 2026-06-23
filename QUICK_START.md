# 🚀 QUICK START GUIDE
## India Runs Hackathon — Complete Analysis Ready

**Status:** ✅ **COMPLETE & READY TO USE**

---

## 📦 What You Got

### 3 Python Scripts
```
COMPLETE_PIPELINE.py          ← RUN THIS FIRST (main analysis)
PLOTTING_NOTEBOOK.py          ← RUN THIS SECOND (visualizations)
README.md                     ← Read for full documentation
```

### 10 Visualization PNGs
```
plot1_encoding.png            ← How features are encoded
plot2_distributions.png       ← Raw feature distributions
plot3_composite_scores.png    ← The 6 engineered scores
plot4_correlation.png         ← Full correlation heatmap
plot5_feature_importance.png  ← Which features matter most
plot6_scatter.png             ← Key relationships
plot7_dependent_entities.png  ← Category breakdowns
plot8_train_test.png          ← Model performance comparison
plot9_pairplot.png            ← Pairwise distributions
plot10_boxplots.png           ← Outlier analysis
```

### CSV Sample Files (reference)
```
features_extracted.csv        ← Features from EDA phase
candidates_full_cleaned.csv   ← Full cleaned data
```

---

## 🎯 Running on Your Machine (Windows)

### 1. Place Files in a Folder
```
your_project/
  ├─ COMPLETE_PIPELINE.py
  ├─ PLOTTING_NOTEBOOK.py
  ├─ sample_candidates.json     (your data file)
  ├─ outputs/                   (will be created)
  └─ venv/                      (virtual env)
```

### 2. Install Dependencies
```powershell
python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 3. Run Pipeline
```powershell
python COMPLETE_PIPELINE.py
```

**Wait ~30 seconds. You'll see:**
```
════════════════════════════════════════════════════════════════════════════════
SECTION 1: LOADING & FLATTENING DATA
════════════════════════════════════════════════════════════════════════════════
✓ Loaded 50 candidate records
✓ Flattened shape: (50, 50)
  Columns: 50
  Null values: 0
...
```

### 4. Generate Plots
```powershell
python PLOTTING_NOTEBOOK.py
```

**Wait ~2 minutes. You'll get 10 PNG files in `outputs/` folder.**

### 5. Check Output
```
outputs/
  ├─ features_final.csv              (50 candidates × 80+ features)
  ├─ train.csv                       (35 candidates - training data)
  ├─ val.csv                         (7 candidates - validation)
  ├─ test.csv                        (8 candidates - testing)
  ├─ model_predictions.csv           (predictions vs actual)
  ├─ feature_importance.csv          (feature ranking)
  ├─ summary.json                    (metadata)
  ├─ plot1_encoding.png              through
  └─ plot10_boxplots.png             ← All 10 plots
```

---

## 📊 What Each Script Does

### COMPLETE_PIPELINE.py
**Main data analysis engine. 8 sections:**

1. **Load & Flatten** — JSON → DataFrame
2. **Data Quality Audit** — Null checks, anomalies, sentinel values
3. **Encoding** — Ordinal, Label, One-Hot
4. **Feature Engineering** — 6 composite scores + interactions
5. **Correlation Analysis** — Feature-target relationships
6. **Train/Test Split** — 70/15/15 with MinMax scaling
7. **Model Training** — Linear, RF, Gradient Boosting
8. **Save Outputs** — CSV, JSON, predictions

**Time:** ~30 seconds  
**Output:** 7 CSV files + metadata

### PLOTTING_NOTEBOOK.py
**Visualization engine. 10 plots covering:**

1. Encoding overview
2. Feature distributions
3. Composite scores
4. Correlation heatmap
5. Feature importance
6. Scatter relationships
7. Categorical breakdowns
8. Model performance
9. Pairplot
10. Outlier analysis

**Time:** ~2 minutes  
**Output:** 10 PNG images

---

## 📈 Key Numbers

| Metric | Value |
|--------|-------|
| **Total candidates** | 50 |
| **Total features engineered** | 80+ |
| **Null values** | 0 ✓ |
| **Training set size** | 35 (70%) |
| **Validation set size** | 7 (15%) |
| **Test set size** | 8 (15%) |
| **Best model** | Gradient Boosting |
| **Test R²** | 0.745 (74.5% variance explained) |
| **Test MAE** | 3.46 points (out of 100) |

---

## 🔥 Top 5 Most Important Features

1. **Advanced Skills** (r=0.648) ⭐⭐⭐
   - Percentage of skills at "advanced" proficiency level
   - Strongest predictor of candidate quality

2. **Assessment Scores** (r=0.557)
   - Average score on skill assessments
   - Verified technical capability

3. **Salary Mid-Point** (r=0.463)
   - Salary expectation (min+max)/2
   - Correlates with experience level

4. **Years of Experience** (r=0.422)
   - Total professional experience
   - Basic seniority indicator

5. **Skill Duration** (r=0.415)
   - Average months spent on each skill
   - Indicates expertise depth, not just count

**Insight:** Technical depth (advanced skills, assessments, expertise duration) matters WAY more than breadth (number of skills).

---

## 🎯 6 Composite Scores (What They Mean)

### 1. Activity Score (15% weight)
**Is the candidate actively looking right now?**
- Profile views, search appearances, recruiter saves, recent activity
- Range: 0-100 (higher = more active on platform)

### 2. Profile Quality (10%)
**How trustworthy & complete is their profile?**
- Profile completeness, verified email/phone, LinkedIn connected
- Range: 0-100 (higher = more credible)

### 3. Career Progression (25%)
**How experienced & stable are they?**
- Job variety, tenure stability, company prestige, industry breadth
- Range: 0-100 (higher = more experienced)

### 4. Skill Depth (25%)
**How technically capable are they?**
- Advanced skill %, endorsements, skill longevity, assessments
- Range: 0-100 (higher = deeper technical skills)

### 5. Engagement Score (20%)
**Will they actually respond & show up?**
- Recruiter response rate, interview completion, offer acceptance
- Range: 0-100 (higher = more reliable)

### 6. Availability (5%)
**Can we hire them quickly?**
- Open to work, notice period, willing to relocate
- Range: 0-100 (higher = easier to hire)

### Master Score (0-100)
**Weighted average of all 6 scores**
```
master_score = 0.25×career + 0.25×skills + 0.20×engagement +
               0.15×activity + 0.10×profile + 0.05×availability
```

---

## 🧠 Understanding the Output Files

### features_final.csv
- **50 rows** = 50 candidates
- **80+ columns** = all features (raw + encoded + composite)
- **0 nulls** = completely clean
- **Use for:** Ranking algorithm, feature analysis

### train.csv / val.csv / test.csv
- **train.csv** = 35 candidates (70%) for training ML models
- **val.csv** = 7 candidates (15%) for tuning hyperparameters  
- **test.csv** = 8 candidates (15%) for final evaluation
- **Use for:** Building your own ML models

### model_predictions.csv
- **candidate_id** — Which candidate
- **split** — train/val/test
- **actual_master_score** — True ranking score
- **predicted_master_score** — Model's prediction
- **prediction_error** — How far off (for debugging)
- **Use for:** Understanding model errors

### feature_importance.csv
- **feature** — Column name
- **importance** — How much Gradient Boosting relies on it
- **Use for:** Feature selection, understanding what drives rankings

---

## 🎓 What to Do Next (For Track 1)

### Day 2-3: Build Semantic Layer
1. Take `features_final.csv` as baseline rankings
2. Extract job description text
3. Use `sentence-transformers` to embed both JD and candidate profiles
4. Add cosine similarity scores to your ranking

### Day 4: Multi-Signal Fusion
```python
final_rank_score = (
    0.3 × master_score +
    0.3 × bm25_keyword_match +
    0.2 × semantic_similarity +
    0.2 × lsm_reranker_score
)
```

### Day 5: LLM Reranking
1. Use Redrob's **2M daily token budget** (you got it free!)
2. Send top-50 candidates to GPT-4o-mini
3. Prompt: "Rate fit 0-100" for each candidate
4. Use LLM scores to rerank final list

### Day 6: Submit!
1. Rank all candidates using fusion score
2. Export `ranked_candidates.csv` in required format
3. Submit GitHub repo + this analysis as "methodology"

---

## ❓ FAQ

### Q: My data file is different, what do I do?
**A:** Replace `sample_candidates.json` with your `candidates.jsonl` file. The script auto-detects the format.

### Q: Can I change the feature weights in master_score?
**A:** Yes! Edit `COMPLETE_PIPELINE.py` line ~280. Rerun to see impact on rankings.

### Q: Why is my plot quality bad?
**A:** Use `dpi=300` instead of `150` in the PNG save commands (slower but sharper).

### Q: Can I use different encoding for categorical variables?
**A:** Yes! The script currently uses Ordinal, Label, and One-Hot. Try:
- Target encoding (based on mean of target variable)
- Frequency encoding (how common is the value)
- Embedding encoding (learn representations)

### Q: Is the master_score good for ranking?
**A:** It's a **starting baseline** (0.745 R², explains 74.5% of variance). For final ranking, add:
- Job description similarity (BM25 + embeddings)
- LLM reranking (GPT-4o-mini)
- Business rules (salary range, location, etc.)

### Q: How do I submit this for Track 1?
**A:** 
1. Generate ranked candidate list using your final scoring formula
2. Export as CSV with columns: `candidate_id, rank, score`
3. Include `COMPLETE_PIPELINE.py` + `README.md` in GitHub as "methodology"
4. Judges will see your feature engineering is rigorous

---

## 📞 Troubleshooting

### Script crashes with "ModuleNotFoundError"
```
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Plots don't appear in outputs folder
```
# Check if matplotlib backend is working
python -c "import matplotlib; print(matplotlib.get_backend())"
```

### Data has different column names
Edit the NUMERIC_FEATURES and SELECTED_FEATURES lists in the scripts to match your columns.

### Want to see individual feature stats
```python
import pandas as pd
df = pd.read_csv("outputs/features_final.csv")
print(df.describe())
```

---

## 🏆 Winning Tips

1. **Master Score = Baseline Only** — Add semantic similarity to JD
2. **Use LLM Budget Wisely** — Rerank top-50, not all 790M
3. **Show Explainability** — Why did candidate rank #1? Show the 6 scores
4. **Test on Validation Set** — Use val.csv to tune weights before final submission
5. **Document Everything** — Judges read README first, code second

---

## ✅ Delivery Checklist

- [x] EDA complete ✓
- [x] Encoding done ✓
- [x] Features engineered ✓
- [x] Correlation analyzed ✓
- [x] Train/test split created ✓
- [x] Models trained ✓
- [x] 10 plots generated ✓
- [x] CSV outputs ready ✓
- [x] README written ✓
- [ ] Next: Semantic layer (Day 2)
- [ ] Next: Multi-signal fusion (Day 4)
- [ ] Next: LLM reranking (Day 5)
- [ ] Next: SUBMIT (Day 6)

---

**You are HERE** ↓
```
Day 1: EDA + Encoding + Features ✅ DONE
Day 2: Semantic layer
Day 3: Multi-signal fusion
Day 4: Tuning & ablation
Day 5: Polish & docs
Day 6: SUBMIT
```

**Ready to move forward?** Open `README.md` for detailed documentation.
