import os
import json
import time
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from build_rag import retrieve
from rag_answer import answer as rag_answer

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
judge = genai.GenerativeModel("gemini-3.5-flash-lite")

# Trimmed test set to stay within free-tier limits (2 calls each)
TEST_QUESTIONS = [
    "how do I cancel my order?",
    "where is my package?",
    "how do I get a refund?",
    "how do I contact customer support?",
]

# ---------------------------------------------------------------
# Retry-with-backoff: if we hit a rate limit (429), wait and retry
# instead of crashing. Standard way to handle production API limits.
# ---------------------------------------------------------------
def call_with_backoff(fn, *args, max_retries=4):
    delay = 20
    for attempt in range(max_retries):
        try:
            return fn(*args)
        except ResourceExhausted:
            if attempt == max_retries - 1:
                raise
            print(f"  rate limited, waiting {delay}s...")
            time.sleep(delay)
            delay *= 2   # exponential backoff
    return None


def judge_answer(question, context, answer_text):
    prompt = f"""You are an evaluator scoring a support assistant's answer.

Question: {question}

Retrieved context the answer should be based on:
{context}

The answer given:
{answer_text}

Score two things from 1 to 5:
- "faithfulness": Is every claim in the answer supported by the retrieved context?
  5 = fully grounded, nothing invented. 1 = mostly made up.
- "relevancy": Does the answer actually address the question?
  5 = directly answers it. 1 = off-topic.

Reply with ONLY a JSON object, no other text:
{{"faithfulness": <1-5>, "relevancy": <1-5>}}"""
    raw = judge.generate_content(prompt).text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


if __name__ == "__main__":
    results = []
    print("Evaluating RAG on", len(TEST_QUESTIONS), "questions...\n")

    for q in TEST_QUESTIONS:
        matches = retrieve(q, k=3)
        context = "\n\n".join(m["answer"] for m in matches)
        ans = call_with_backoff(rag_answer, q)
        scores = call_with_backoff(judge_answer, q, context, ans)
        results.append(scores)
        print(f"F={scores['faithfulness']} R={scores['relevancy']}  {q}")
        time.sleep(5)   # gentle pacing between questions

    n = len(results)
    avg_f = sum(r["faithfulness"] for r in results) / n
    avg_r = sum(r["relevancy"] for r in results) / n
    print("\n" + "=" * 40)
    print(f"Average faithfulness: {avg_f:.2f} / 5")
    print(f"Average relevancy:    {avg_r:.2f} / 5")
    print("=" * 40)
