import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# STEP 1: Load the data we saved on Day 1
# ---------------------------------------------------------------
df = pd.read_csv("data/tickets.csv")

# The dataset is big (26k rows). For building/testing, use a slice
# so embedding is fast. We'll scale up later once it works.
df = df.reset_index(drop=True)

# We embed the customer QUESTION (instruction). When a new question
# comes in, we find the closest past questions and return THEIR responses.
questions = df["instruction"].tolist()
answers = df["response"].tolist()

# ---------------------------------------------------------------
# STEP 2: Load the embedding model
# all-MiniLM-L6-v2 is small, fast, and turns text into 384 numbers.
# First run downloads it (~90MB), then it's cached.
# ---------------------------------------------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------
# STEP 3: Embed every question -> a matrix of shape (2000, 384)
# Each row is one question turned into 384 numbers.
# ---------------------------------------------------------------
print("Embedding questions (this takes a minute)...")
embeddings = model.encode(questions, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")  # FAISS needs float32

# ---------------------------------------------------------------
# STEP 4: Build the FAISS index
# IndexFlatL2 = measures straight-line distance between vectors.
# Smaller distance = more similar meaning.
# ---------------------------------------------------------------
dimension = embeddings.shape[1]          # 384
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)                    # load all vectors into the index
print(f"Index built with {index.ntotal} vectors.")

# ---------------------------------------------------------------
# STEP 5: Search! Define a function that retrieves for any question.
# ---------------------------------------------------------------
def retrieve(query, k=3):
    """Return the top-k most similar past Q&As for a new query."""
    # Embed the incoming question the SAME way as the stored ones
    q_vec = model.encode([query]).astype("float32")
    # Ask FAISS for the k nearest stored vectors
    distances, indices = index.search(q_vec, k)
    # indices[0] holds the row numbers of the closest matches
    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "distance": float(distances[0][rank]),
            "matched_question": questions[idx],
            "answer": answers[idx],
        })
    return results

# ---------------------------------------------------------------
# STEP 6: Try it out
# ---------------------------------------------------------------
if __name__ == "__main__":
    test_query = "how do I cancel my order?"
    print(f"\nQuery: {test_query}\n")
    for r in retrieve(test_query):
        print(f"[{r['rank']}] distance={r['distance']:.2f}")
        print(f"    matched Q: {r['matched_question']}")
        print(f"    answer: {r['answer'][:120]}...")
        print()