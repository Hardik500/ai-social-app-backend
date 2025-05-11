import json
import re
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
    def _parse_json_response(self, content: Any) -> Dict[str, Any]:
        """
        Utility method to parse and clean JSON content from model responses.
        
        Args:
            content: Raw content from the model, could be string or dict
            
        Returns:
            Parsed JSON data as a dictionary
            
        Raises:
            Exception: If JSON parsing fails
        """
        if not isinstance(content, str):
            return content
            
        # Try to clean/fix common JSON issues
        cleaned_content = content.strip()
        
        # If content starts with ``` and ends with ```, strip those
        if cleaned_content.startswith("```json") and cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[7:-3].strip()
        elif cleaned_content.startswith("```") and cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[3:-3].strip()
        
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            # Last resort - try to find JSON-like structure with regex
            json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
            if json_match:
                potential_json = json_match.group(0)
                return json.loads(potential_json)
            raise  # Re-raise if no JSON-like structure found
    
    async def validate_response(self, question: str, response: str, personality_context: Dict[str, Any], previous_exchanges: Optional[List[Dict[str, str]]] = None, preferred_communication_style: Optional[str] = None) -> ValidationResult:
        prompt = prompt_manager.format_template(
            "response_validation",
            question=question,
            response=response,
            personality_context=json.dumps(personality_context),
            previous_exchanges=json.dumps(previous_exchanges) if previous_exchanges else "",
            preferred_communication_style=preferred_communication_style or ""
        )
        result = await model_provider.generate_chat(
            [{"role": "system", "content": prompt}],
            format_json=True
        )
        if "message" in result and "content" in result["message"]:
            try:
                content = result["message"]["content"]
                data = self._parse_json_response(content)
                return ValidationResult.from_dict(data)
            except Exception as e:
                return ValidationResult(False, 0.0, [f"Parsing error: {str(e)}"], False, None, "low")
        return ValidationResult(False, 0.0, ["No response from model"], False, None, "low")

    async def generate_followup(self, question: str, response: str, personality_traits: Dict[str, Any], engagement_level: str, previous_exchanges: Optional[List[Dict[str, str]]] = None, preferred_communication_style: Optional[str] = None) -> Dict[str, str]:
        prompt = prompt_manager.format_template(
            "followup_generation",
            question=question,
            response=response,
            personality_traits=json.dumps(personality_traits),
            engagement_level=engagement_level,
            previous_exchanges=json.dumps(previous_exchanges) if previous_exchanges else "[]",
            preferred_communication_style=preferred_communication_style or "casual"
        )
        result = await model_provider.generate_chat(
            [{"role": "system", "content": prompt}],
            format_json=True
        )
        if "message" in result and "content" in result["message"]:
            try:
                content = result["message"]["content"]
                return self._parse_json_response(content)
            except Exception as e:
                return {"followup_question": None, "reasoning": f"Parsing error: {str(e)}"}
        return {"followup_question": None, "reasoning": "No response from model"}

response_validator = ResponseValidator() 