import os
import joblib
from typing import TypedDict
from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph, END

# Reuse what you already built
from rag_answer import answer as rag_answer

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
router_model = genai.GenerativeModel("gemini-3.5-flash-lite")

# Load the classifier you trained on Day 5
classifier = joblib.load("data/classifier.joblib")


# ---------------------------------------------------------------
# STATE: the shared dictionary that flows through the graph.
# ---------------------------------------------------------------
class AgentState(TypedDict):
    question: str      # the user's input
    route: str         # "rag" or "classify" (set by the router)
    answer: str        # the final response (set by a tool node)


# ---------------------------------------------------------------
# NODE 1: the router. Reads question, writes route.
# ---------------------------------------------------------------
def router_node(state: AgentState) -> AgentState:
    prompt = f"""You are a router. Decide which tool handles the user's message.
Tools:
- "rag": support questions the user wants answered.
- "classify": only when the user explicitly wants the CATEGORY of a ticket.
Reply with ONLY one word: rag or classify.

User message: {state['question']}
Answer:"""
    decision = router_model.generate_content(prompt).text.strip().lower()
    state["route"] = "classify" if "classify" in decision else "rag"
    print(f"[router decided: {state['route']}]")
    return state


# ---------------------------------------------------------------
# NODE 2: RAG tool. Answers support questions.
# ---------------------------------------------------------------
def rag_node(state: AgentState) -> AgentState:
    state["answer"] = rag_answer(state["question"])
    return state


# ---------------------------------------------------------------
# NODE 3: classifier tool. Predicts the ticket category.
# ---------------------------------------------------------------
def classify_node(state: AgentState) -> AgentState:
    category = classifier.predict([state["question"]])[0]
    state["answer"] = f"That ticket falls under the category: {category}"
    return state


# ---------------------------------------------------------------
# CONDITIONAL EDGE: reads state["route"], picks the next node.
# ---------------------------------------------------------------
def choose_tool(state: AgentState) -> str:
    return state["route"]


# ---------------------------------------------------------------
# BUILD THE GRAPH: connect the nodes with edges.
# ---------------------------------------------------------------
graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("rag", rag_node)
graph.add_node("classify", classify_node)

graph.set_entry_point("router")           # start at the router
graph.add_conditional_edges(              # router branches to a tool
    "router", choose_tool,
    {"rag": "rag", "classify": "classify"},
)
graph.add_edge("rag", END)                # tools finish the run
graph.add_edge("classify", END)

app = graph.compile()


# ---------------------------------------------------------------
# INTERACTIVE LOOP
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Agent ready. Ask a support question, or ask it to")
    print("classify a ticket. Type 'quit' to stop.")
    print("=" * 50 + "\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", ""):
            print("Goodbye!")
            break
        result = app.invoke({"question": q, "route": "", "answer": ""})
        print("\nAgent:", result["answer"], "\n")
