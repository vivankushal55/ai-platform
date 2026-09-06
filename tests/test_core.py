import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from build_rag import retrieve
from agent import app as agent


def _ask(question):
    """Run a question through the agent, return the full result."""
    return agent.invoke({"question": question, "route": "", "answer": ""})


def test_retrieve_returns_k_results():
    """Retrieval should return exactly k results."""
    results = retrieve("how do I cancel my order?", k=3)
    assert len(results) == 3


def test_retrieve_finds_relevant_match():
    """A cancel query should surface a cancel-related result with low distance."""
    results = retrieve("how do I cancel my order?", k=3)
    assert results[0]["distance"] < 1.0
    assert "cancel" in results[0]["matched_question"].lower()


def test_agent_routes_support_question_to_rag():
    """A plain support question should route to rag."""
    result = _ask("how do I get a refund?")
    assert result["route"] == "rag"
    assert len(result["answer"]) > 0


def test_agent_routes_classify_request_to_classify():
    """An explicit classify request should route to classify."""
    result = _ask("what category is this ticket: my card was declined")
    assert result["route"] == "classify"
