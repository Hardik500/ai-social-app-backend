import json
from typing import Dict, Any, List, Optional
from app.core.prompt_manager import prompt_manager
from app.services.model_provider import model_provider

class ValidationResult:
    def __init__(self, is_valid: bool, score: float, issues: List[str], needs_followup: bool, followup_suggestion: Optional[str], engagement_level: str):
        self.is_valid = is_valid
        self.score = score
        self.issues = issues
        self.needs_followup = needs_followup
        self.followup_suggestion = followup_suggestion
        self.engagement_level = engagement_level

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            is_valid=d.get("is_valid", False),
            score=d.get("score", 0.0),
            issues=d.get("issues", []),
            needs_followup=d.get("needs_followup", False),
            followup_suggestion=d.get("followup_suggestion"),
            engagement_level=d.get("engagement_level", "low")
        )

class ResponseValidator:
    async def validate_response(self, question: str, response: str, personality_context: Dict[str, Any]) -> ValidationResult:
        prompt = prompt_manager.format_template(
            "response_validation",
            question=question,
            response=response,
            personality_context=json.dumps(personality_context)
        )
        result = await model_provider.generate_chat(
            [{"role": "system", "content": prompt}],
            format_json=True
        )
        if "message" in result and "content" in result["message"]:
            try:
                content = result["message"]["content"]
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content
                return ValidationResult.from_dict(data)
            except Exception as e:
                return ValidationResult(False, 0.0, [f"Parsing error: {str(e)}"], False, None, "low")
        return ValidationResult(False, 0.0, ["No response from model"], False, None, "low")

    async def generate_followup(self, question: str, response: str, personality_traits: Dict[str, Any], engagement_level: str) -> Dict[str, str]:
        prompt = prompt_manager.format_template(
            "followup_generation",
            question=question,
            response=response,
            personality_traits=json.dumps(personality_traits),
            engagement_level=engagement_level
        )
        result = await model_provider.generate_chat(
            [{"role": "system", "content": prompt}],
            format_json=True
        )
        if "message" in result and "content" in result["message"]:
            try:
                content = result["message"]["content"]
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content
                return data
            except Exception as e:
                return {"followup_question": None, "reasoning": f"Parsing error: {str(e)}"}
        return {"followup_question": None, "reasoning": "No response from model"}

response_validator = ResponseValidator() 