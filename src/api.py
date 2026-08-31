from fastapi import FastAPI
from pydantic import BaseModel

# Reuse your compiled LangGraph agent
from agent import app as agent

# ---------------------------------------------------------------
# The API application
# ---------------------------------------------------------------
api = FastAPI(title="Support Intelligence Platform")


# ---------------------------------------------------------------
# Request shape: the caller sends JSON like {"question": "..."}
# Pydantic validates it automatically.
# ---------------------------------------------------------------
class Query(BaseModel):
    question: str


# ---------------------------------------------------------------
# A simple health check — visit the root URL to confirm it's alive
# ---------------------------------------------------------------
@api.get("/")
def health():
    return {"status": "ok", "service": "Support Intelligence Platform"}


# ---------------------------------------------------------------
# The main endpoint: POST a question, get the agent's answer +
# which route it chose (rag or classify).
# ---------------------------------------------------------------
@api.post("/ask")
def ask(query: Query):
    result = agent.invoke({
        "question": query.question,
        "route": "",
        "answer": "",
    })
    return {
        "question": query.question,
        "route": result["route"],
        "answer": result["answer"],
    }
