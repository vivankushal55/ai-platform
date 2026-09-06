import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from build_rag import retrieve

# Only import the agent when we actually need it (keeps CI light)
def _ask(question):
    from agent import app as agent
    return agent.invoke({"question": question, "route": "", "answer": ""})


def test_retrieve_returns_k_results():
    results = retrieve("how do I cancel my order?", k=3)
    assert len(results) == 3


def test_retrieve_finds_relevant_match():
    results = retrieve("how do I cancel my order?", k=3)
    assert results[0]["distance"] < 1.0
    assert "cancel" in results[0]["matched_question"].lower()


@pytest.mark.api
def test_agent_routes_support_question_to_rag():
    result = _ask("how do I get a refund?")
    assert result["route"] == "rag"
    assert len(result["answer"]) > 0


@pytest.mark.api
def test_agent_routes_classify_request_to_classify():
    result = _ask("what category is this ticket: my card was declined")
    assert result["route"] == "classify"
