import os
from dotenv import load_dotenv
import google.generativeai as genai

# Reuse the retrieval engine you built on Day 2
from build_rag import retrieve

# ---------------------------------------------------------------
# Load the Gemini API key from .env
# ---------------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.6-flash")

# ---------------------------------------------------------------
# RAG: give the LLM ONLY the retrieved answers, instruct it to
# answer from just those. This is what prevents hallucination.
# ---------------------------------------------------------------
def answer(query, k=3):
    # STEP 1: retrieve top-k relevant past Q&As (Day 2 code)
    matches = retrieve(query, k=k)

    # STEP 2: format them into a context block
    context = "\n\n".join(
        f"Example {m['rank']}:\nQ: {m['matched_question']}\nA: {m['answer']}"
        for m in matches
    )

    # STEP 3: build the prompt
    prompt = f"""You are a customer support assistant. Answer the user's question
using ONLY the information in the retrieved examples below. Do not invent
policies, numbers, or steps that aren't supported by them. If the examples
don't contain the answer, say you don't have that information.

Retrieved examples:
{context}

User question: {query}

Answer:"""

    # STEP 4: call Gemini
    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    q = "how do I cancel my order?"
    print(f"Q: {q}\n")
    print(answer(q))