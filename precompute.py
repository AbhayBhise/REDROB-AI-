import os, json, argparse
import numpy as np

os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def build_candidate_text(c):
    p = c.get('profile', {})
    parts = []
    if p.get('headline'): parts.append(p['headline'])
    if p.get('summary'):  parts.append(p['summary'])
    if p.get('current_title'): parts.append(p['current_title'])
    if p.get('current_industry'): parts.append(p['current_industry'])
    skill_parts = []
    for s in c.get('skills', []):
        name = s.get('name', '')
        prof = s.get('proficiency', '')
        if prof == 'advanced': skill_parts.append(f"{name} {name} {name}")
        elif prof == 'intermediate': skill_parts.append(f"{name} {name}")
        else: skill_parts.append(name)
    if skill_parts: parts.append('Skills: ' + ', '.join(skill_parts))
    for job in c.get('career_history', []):
        if job.get('title'): parts.append(job['title'])
        if job.get('description'): parts.append(job['description'])
    for edu in c.get('education', []):
        if edu.get('field_of_study'): parts.append(edu['field_of_study'])
    return ' '.join(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='candidates.jsonl')
    parser.add_argument('--out-embeddings', default='data/processed/candidates_embeddings.npz')
    parser.add_argument('--out-ids', default='data/candidate_ids_ordered.json')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    print("Loading candidates...")
    candidates = []
    with open(args.candidates, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates")

    texts = [build_candidate_text(c) for c in candidates]
    ids   = [c['candidate_id'] for c in candidates]

    print("Loading BAAI/bge-base-en-v1.5 via transformers (768-dim embeddings)...")
    from transformers import AutoTokenizer, AutoModel
    import torch

    model_name = 'BAAI/bge-base-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    print("Model loaded.")

    print(f"Embedding {len(texts):,} candidates in batches of {args.batch_size}...")
    all_embeddings = []

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors='pt'
        )
        with torch.no_grad():
            output = model(**encoded)
            emb = output.last_hidden_state[:, 0, :]
            emb = emb / emb.norm(dim=1, keepdim=True)
        all_embeddings.append(emb.numpy())

        if i % 10000 == 0:
            pct = (i / len(texts)) * 100
            print(f"  Progress: {i:,}/{len(texts):,} ({pct:.1f}%)")

    embeddings = np.vstack(all_embeddings)
    print(f"Embedding shape: {embeddings.shape}")

    # Convert to float16 for compression (saves ~50% space while maintaining quality)
    embeddings_float16 = embeddings.astype(np.float16)
    
    print(f"Saving compressed embeddings to {args.out_embeddings}...")
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.out_embeddings), exist_ok=True)
    np.savez_compressed(args.out_embeddings, embeddings=embeddings_float16)

    print(f"Saving ID mapping to {args.out_ids}...")
    with open(args.out_ids, 'w') as f:
        json.dump(ids, f)

    size_mb = os.path.getsize(args.out_embeddings) / (1024 * 1024)
    print(f"\nDone!")
    print(f"  Embeddings: {args.out_embeddings} ({size_mb:.1f} MB, compressed .npz format)")
    print(f"  Original shape: {embeddings.shape}")
    print(f"  Stored as: float16 (50% size reduction from float32)")
    print(f"  IDs saved: {args.out_ids}")
    print(f"\nNow run: python rank.py --candidates candidates.jsonl --out submission.csv")

if __name__ == '__main__':
    main()
