import os
from dotenv import load_dotenv
import google.generativeai as genai

# Reuse the retrieval engine you built on Day 2
from build_rag import retrieve

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.6-flash")

def answer(query, k=3):
    matches = retrieve(query, k=k)
    context = "\n\n".join(
        f"Example {m['rank']}:\nQ: {m['matched_question']}\nA: {m['answer']}"
        for m in matches
    )
    prompt = f"""You are a customer support assistant. Answer the user's question
using ONLY the information in the retrieved examples below. Do not invent
policies, numbers, or steps that aren't supported by them. If the examples
don't contain the answer, say you don't have that information.

Retrieved examples:
{context}

User question: {query}

Answer:"""
    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Support Assistant ready. Type your question.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 50 + "\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", ""):
            print("Goodbye!")
            break
        print("\nAssistant:", answer(query), "\n")
