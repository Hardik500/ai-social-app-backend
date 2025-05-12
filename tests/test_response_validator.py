import pytest
from unittest import mock
from app.services.response_validator import response_validator, ValidationResult
from app.services.model_provider import ModelProvider

model_provider = None

@pytest.mark.asyncio
async def test_validate_response_success(monkeypatch):
    print("Starting test_validate_response_success")
    # Mock model_provider.generate_chat to return a valid JSON string
    async def mock_generate_chat(messages, system_prompt=None, stream=False, format_json=False):
        print(f"Mocking generate_chat with messages: {messages}")
        return {
            "message": {
                "content": '{"is_valid": true, "score": 0.9, "issues": [], "needs_followup": false, "followup_suggestion": null, "engagement_level": "high"}'
            }
        }
    monkeypatch.setattr("app.services.response_validator.model_provider", mock.Mock(generate_chat=mock_generate_chat))

    answers = [{'content': 'Just chilling bro, you know. Nothing much. Tu bata?', 'type': 'text'}]
    response_text = " ".join([a["content"] for a in answers])
    result = await response_validator.validate_response(
        question="What's up?",
        response=response_text,
        personality_context={"traits": {}, "communication_style": {}, "interests": []},
        previous_exchanges=[],
        preferred_communication_style=None
    )
    assert isinstance(result, ValidationResult)
    assert result.is_valid is True
    assert result.score == 0.9
    assert result.engagement_level == "high"

@pytest.mark.asyncio
async def test_validate_response_parsing_error(monkeypatch):
    print("Starting test_validate_response_parsing_error")
    # Mock model_provider.generate_chat to return code fences
    async def mock_generate_chat(messages, system_prompt=None, stream=False, format_json=False):
        print(f"Mocking generate_chat with messages: {messages}")
        return {
            "message": {
                "content": '```json\n{"is_valid": true, "score": 1.0, "issues": [], "needs_followup": false, "followup_suggestion": null, "engagement_level": "high"}\n```'
            }
        }
    monkeypatch.setattr("app.services.response_validator.model_provider", mock.Mock(generate_chat=mock_generate_chat))

    answers = [{'content': 'Just chilling bro, you know. Nothing much. Tu bata?', 'type': 'text'}]
    response_text = " ".join([a["content"] for a in answers])
    result = await response_validator.validate_response(
        question="What's up?",
        response=response_text,
        personality_context={"traits": {}, "communication_style": {}, "interests": []},
        previous_exchanges=[],
        preferred_communication_style=None
    )
    # Should fail to parse and return is_valid False
    assert isinstance(result, ValidationResult)
    assert result.is_valid is False
    assert any("Parsing error" in issue for issue in result.issues) 
    
def test_pytest_runs():
    print("Pytest is running this file!")
    assert True