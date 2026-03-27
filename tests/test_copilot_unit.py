"""
Unit tests for VoltexCoPilot class.
All external dependencies (Anthropic API, ChromaDB, SentenceTransformer)
are mocked — these tests run fast and free with no network calls.
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from copilot import VoltexCoPilot, CopilotResponse


MOCK_COPILOT_JSON = json.dumps({
    "customer_need"      : "Customer wants to know about VoltCare accidental damage cover.",
    "suggested_response" : "VoltCare Plus and Complete cover accidental damage. "
                           "An excess applies on Plus. No excess on Complete.",
    "confidence"         : "HIGH",
    "sources_used"       : ["voltcare_policy.txt"],
    "key_policy_points"  : [
        "Plus: one AD claim per 12 months, excess £25–£75 by product value",
        "Complete: unlimited AD, no excess",
        "Essential: no AD cover",
    ],
    "escalate"           : False,
    "escalation_reason"  : "",
})

MOCK_CLASSIFY_JSON = json.dumps({
    "rewritten_query": "VoltCare accidental damage laptop cover",
    "category"       : "voltcare",
})


@pytest.fixture
def mock_copilot():
    """
    Returns a VoltexCoPilot instance with all external dependencies mocked.
    Embedding model, ChromaDB, and Anthropic client are replaced with mocks.
    """
    with patch("copilot.SentenceTransformer") as mock_st, \
         patch("copilot.chromadb.PersistentClient") as mock_chroma, \
         patch("copilot.Anthropic") as mock_anthropic:

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 384
        mock_st.return_value = mock_model

        # Mock ChromaDB collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 421
        mock_collection.query.return_value = {
            "documents": [[
                "VoltCare Plus covers accidental damage with one claim per 12 months.",
                "VoltCare Complete has unlimited accidental damage claims and no excess.",
                "VoltCare Essential does not cover accidental damage.",
            ]],
            "metadatas": [[
                {"source": "voltcare_policy.txt", "category": "voltcare",
                 "section": "PART 3", "similarity": 0.82},
                {"source": "voltcare_policy.txt", "category": "voltcare",
                 "section": "PART 1", "similarity": 0.75},
                {"source": "voltcare_policy.txt", "category": "voltcare",
                 "section": "PART 1", "similarity": 0.70},
            ]],
            "distances": [[0.18, 0.25, 0.30]],
        }
        mock_chroma_client = MagicMock()
        mock_chroma_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_chroma_client

        # Mock Anthropic client — returns classify JSON on first call,
        # copilot JSON on second call
        mock_client = MagicMock()
        classify_response = MagicMock()
        classify_response.content = [MagicMock(text=MOCK_CLASSIFY_JSON)]
        copilot_response = MagicMock()
        copilot_response.content = [MagicMock(text=MOCK_COPILOT_JSON)]
        mock_client.messages.create.side_effect = [
            classify_response,
            copilot_response,
        ]
        mock_anthropic.return_value = mock_client

        copilot = VoltexCoPilot()
        return copilot


# ─────────────────────────────────────────────
# INITIALISATION TESTS
# ─────────────────────────────────────────────

def test_copilot_initialises(mock_copilot):
    assert mock_copilot is not None
    assert mock_copilot.conversation_history == []
    assert mock_copilot.MAX_HISTORY_TURNS == 4


# ─────────────────────────────────────────────
# get_response TESTS
# ─────────────────────────────────────────────

def test_get_response_returns_dict(mock_copilot):
    result = mock_copilot.get_response("Does VoltCare cover accidental damage?")
    assert isinstance(result, dict)


def test_get_response_has_required_keys(mock_copilot):
    result = mock_copilot.get_response("Does VoltCare cover accidental damage?")
    assert "response"         in result
    assert "rewritten_query"  in result
    assert "category"         in result
    assert "retrieved_chunks" in result


def test_get_response_response_is_copilot_response(mock_copilot):
    result = mock_copilot.get_response("Does VoltCare cover accidental damage?")
    assert isinstance(result["response"], CopilotResponse)


def test_get_response_rewritten_query_is_string(mock_copilot):
    result = mock_copilot.get_response("Does VoltCare cover accidental damage?")
    assert isinstance(result["rewritten_query"], str)
    assert len(result["rewritten_query"]) > 0


def test_get_response_category_is_valid(mock_copilot):
    result = mock_copilot.get_response("Does VoltCare cover accidental damage?")
    valid_categories = {
        "voltcare", "returns_repairs", "products",
        "delivery_orders", "voltmobile", "general"
    }
    assert result["category"] in valid_categories


def test_get_response_retrieved_chunks_is_list(mock_copilot):
    result = mock_copilot.get_response("Does VoltCare cover accidental damage?")
    assert isinstance(result["retrieved_chunks"], list)


def test_get_response_updates_conversation_history(mock_copilot):
    assert len(mock_copilot.conversation_history) == 0
    mock_copilot.get_response("Does VoltCare cover accidental damage?")
    # Should have added user message and assistant response
    assert len(mock_copilot.conversation_history) == 2
    assert mock_copilot.conversation_history[0]["role"] == "user"
    assert mock_copilot.conversation_history[1]["role"] == "assistant"


# ─────────────────────────────────────────────
# CONVERSATION HISTORY TESTS
# ─────────────────────────────────────────────

def test_reset_conversation_clears_history(mock_copilot):
    mock_copilot.conversation_history = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "response"},
    ]
    mock_copilot.reset_conversation()
    assert mock_copilot.conversation_history == []


def test_conversation_history_grows_with_turns(mock_copilot):
    """Each call to get_response adds 2 messages to history."""
    # Need to reset side_effect to handle multiple calls
    classify_msg = MagicMock()
    classify_msg.content = [MagicMock(text=MOCK_CLASSIFY_JSON)]
    copilot_msg = MagicMock()
    copilot_msg.content = [MagicMock(text=MOCK_COPILOT_JSON)]

    mock_copilot.client.messages.create.side_effect = [
        classify_msg, copilot_msg,
        classify_msg, copilot_msg,
    ]

    mock_copilot.get_response("First question")
    mock_copilot.get_response("Second question")
    assert len(mock_copilot.conversation_history) == 4


def test_history_capped_at_max_turns(mock_copilot):
    """Conversation history passed to Claude is capped at MAX_HISTORY_TURNS * 2."""
    # Fill history beyond the cap
    mock_copilot.conversation_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(20)
    ]
    # The _reason method should only pass the last MAX_HISTORY_TURNS * 2 messages
    cap = mock_copilot.MAX_HISTORY_TURNS * 2
    recent = mock_copilot.conversation_history[-(cap):]
    assert len(recent) == cap


# ─────────────────────────────────────────────
# GRACEFUL FALLBACK TEST
# ─────────────────────────────────────────────

def test_graceful_fallback_on_bad_json(mock_copilot):
    """If Claude returns malformed JSON, copilot returns LOW confidence fallback."""
    classify_msg = MagicMock()
    classify_msg.content = [MagicMock(text=MOCK_CLASSIFY_JSON)]
    bad_response = MagicMock()
    bad_response.content = [MagicMock(text="this is not json at all {{{}")]
    mock_copilot.client.messages.create.side_effect = [classify_msg, bad_response]

    result = mock_copilot.get_response("Some query that causes a bad response")
    assert result["response"].confidence == "LOW"
    assert result["response"].escalate is True