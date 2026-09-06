import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Anchor all paths to the project root, so this works no matter
# where the script is launched from (terminal, uvicorn, Docker).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss.index")
DATA_PATH = os.path.join(BASE_DIR, "data", "tickets.csv")
# Fallback to the small sample (used in CI where the full dataset isn't committed)
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "tests", "sample_tickets.csv")

# Load data (same order every time, so the index rows line up)
df = pd.read_csv(DATA_PATH).reset_index(drop=True)
questions = df["instruction"].tolist()
answers = df["response"].tolist()

# Model is needed to embed incoming queries (fast to load)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Build the index ONCE, then reuse the saved copy
if os.path.exists(INDEX_PATH):
    print("Loading cached FAISS index...")
    index = faiss.read_index(INDEX_PATH)
else:
    print("Embedding 26k questions (one-time, ~1-2 min)...")
    embeddings = model.encode(questions, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print("Saved index to disk.")

print(f"Index ready with {index.ntotal} vectors.")


def retrieve(query, k=3):
    q_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, k)
    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "distance": float(distances[0][rank]),
            "matched_question": questions[idx],
            "answer": answers[idx],
        })
    return results


if __name__ == "__main__":
    for r in retrieve("how do I cancel my order?"):
        print(f"[{r['rank']}] {r['matched_question']}")
