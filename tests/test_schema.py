"""
Tests for the CopilotResponse Pydantic schema.
Verifies the schema validates correctly and rejects invalid inputs.
No API calls — pure schema validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pydantic import ValidationError
from copilot import CopilotResponse


VALID_PAYLOAD = {
    "customer_need"      : "Customer wants to know if VoltCare covers accidental damage.",
    "suggested_response" : "Yes, accidental damage is covered under VoltCare Plus and Complete. "
                           "An excess applies on Plus claims. Complete has no excess.",
    "confidence"         : "HIGH",
    "sources_used"       : ["voltcare_policy.txt"],
    "key_policy_points"  : [
        "Plus tier: one AD claim per 12 months, excess applies",
        "Complete tier: unlimited AD claims, no excess",
        "Essential tier: AD not covered",
    ],
    "escalate"           : False,
    "escalation_reason"  : "",
}


def test_valid_payload_creates_response():
    response = CopilotResponse(**VALID_PAYLOAD)
    assert response.customer_need == VALID_PAYLOAD["customer_need"]
    assert response.confidence == "HIGH"
    assert response.escalate is False


def test_all_confidence_levels_accepted():
    for level in ["HIGH", "MEDIUM", "LOW"]:
        payload = {**VALID_PAYLOAD, "confidence": level}
        response = CopilotResponse(**payload)
        assert response.confidence == level


def test_invalid_confidence_rejected():
    with pytest.raises(ValidationError):
        CopilotResponse(**{**VALID_PAYLOAD, "confidence": "VERY_HIGH"})


def test_escalate_true_with_reason():
    payload = {
        **VALID_PAYLOAD,
        "escalate"          : True,
        "escalation_reason" : "Customer has 3+ claims in 24 months — requires TL authorisation.",
    }
    response = CopilotResponse(**payload)
    assert response.escalate is True
    assert "TL authorisation" in response.escalation_reason


def test_sources_used_is_list():
    response = CopilotResponse(**VALID_PAYLOAD)
    assert isinstance(response.sources_used, list)


def test_sources_used_can_be_empty():
    payload = {**VALID_PAYLOAD, "sources_used": [], "confidence": "LOW"}
    response = CopilotResponse(**payload)
    assert response.sources_used == []


def test_key_policy_points_is_list():
    response = CopilotResponse(**VALID_PAYLOAD)
    assert isinstance(response.key_policy_points, list)
    assert len(response.key_policy_points) >= 1


def test_missing_required_field_rejected():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "customer_need"}
    with pytest.raises(ValidationError):
        CopilotResponse(**payload)


def test_suggested_response_is_string():
    response = CopilotResponse(**VALID_PAYLOAD)
    assert isinstance(response.suggested_response, str)
    assert len(response.suggested_response) > 0


def test_multi_source_response():
    payload = {
        **VALID_PAYLOAD,
        "sources_used": ["voltcare_policy.txt", "repairs_returns_policy.txt"],
        "confidence"  : "MEDIUM",
    }
    response = CopilotResponse(**payload)
    assert len(response.sources_used) == 2
    assert "voltcare_policy.txt" in response.sources_used