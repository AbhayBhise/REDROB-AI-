"""
sandbox_app.py — Interactive demo for India Runs Track 1 submission
Run with: streamlit run sandbox_app.py
"""
import streamlit as st
import json, csv, io, os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rank import compute_score, generate_reasoning

st.set_page_config(
    page_title="Redrob AI — Candidate Ranking",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Candidate Ranking — India Runs Hackathon Track 1")
st.markdown("""
**Multi-signal ranking system** combining:
- 🔍 **Semantic similarity** (BAAI/bge-small-en-v1.5 embeddings)
- 📝 **BM25 keyword overlap**
- 🏆 **Master score** (skills, experience, activity, location, title fit, consulting penalty)

**Fusion:** `0.40 × Semantic + 0.25 × BM25 + 0.35 × Master`
""")

st.divider()

uploaded = st.file_uploader(
    "Upload candidates.jsonl (up to 100 candidates)",
    type=['jsonl', 'json'],
    help="Each line should be a valid JSON candidate object matching the hackathon schema."
)

if uploaded:
    candidates = []
    content = uploaded.read().decode('utf-8')
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    st.info(f"Loaded **{len(candidates)}** candidates from uploaded file.")

    if len(candidates) > 100:
        st.warning("More than 100 candidates uploaded. Only the first 100 will be ranked.")
        candidates = candidates[:100]

    if st.button("🚀 Run Ranking", type="primary"):
        with st.spinner(f"Scoring {len(candidates)} candidates using Master Score (semantic+BM25 requires precomputed embeddings)..."):
            scored = []
            for c in candidates:
                s = compute_score(c)
                scored.append((c, s))
            scored.sort(key=lambda x: (-x[1], x[0]['candidate_id']))

        st.success(f"✅ Ranked {len(scored)} candidates!")

        rows = []
        for rank, (c, score) in enumerate(scored[:100], 1):
            p = c.get('profile', {})
            reasoning = generate_reasoning(c, score, rank)
            rows.append({
                'rank': rank,
                'candidate_id': c['candidate_id'],
                'score': round(score, 4),
                'title': p.get('current_title', 'N/A'),
                'experience': f"{p.get('years_of_experience', 0):.1f}y",
                'location': p.get('location', 'N/A'),
                'reasoning': reasoning
            })

        st.subheader("📊 Top Ranked Candidates")
        st.dataframe(
            rows,
            use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "score": st.column_config.NumberColumn("Score", format="%.4f"),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            }
        )

        # Download button
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                'candidate_id': r['candidate_id'],
                'rank': r['rank'],
                'score': r['score'],
                'reasoning': r['reasoning']
            })

        st.download_button(
            label="⬇️ Download submission.csv",
            data=output.getvalue(),
            file_name="submission.csv",
            mime="text/csv"
        )
else:
    st.markdown("""
### How to use
1. Export a sample of candidates from `candidates.jsonl` (up to 100 lines)
2. Upload the file above
3. Click **Run Ranking**
4. Download the ranked `submission.csv`

### Sample command to extract 100 candidates:
```powershell
Get-Content candidates.jsonl | Select-Object -First 100 | Out-File sample_100.jsonl -Encoding utf8
```
""")

st.divider()
st.caption("Redrob AI — India Runs Hackathon Track 1 | Multi-Signal Candidate Ranking System")
