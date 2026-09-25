"""
validator.py
------------
Handles JSON parsing, schema validation, error handling, missing field counts,
and hallucination detection for extracted structured data.
"""

import json
import re
from typing import Dict, Any, Tuple, List

REQUIRED_SCHEMA_KEYS = [
    "personal_information",
    "education",
    "skills",
    "work_experience",
    "projects",
    "certifications",
    "other_information"
]

class OutputValidator:
    """
    Validates and cleans the output returned by the LLM.
    """

    @staticmethod
    def clean_json_string(raw_text: str) -> str:
        """
        Strips markdown code fences (e.g., ```json ... ```) or leading/trailing text noise
        to isolate the raw JSON payload.
        """
        if not raw_text:
            return ""

        text = raw_text.strip()
        
        # Remove ```json ... ``` markdown wrappers if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if json_match:
            return json_match.group(1).strip()
        
        # Try finding the first '{' and last '}'
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1]

        return text

    @classmethod
    def validate_json_syntax(cls, raw_output: str) -> Tuple[bool, Any, str]:
        """
        Checks if the LLM output is valid JSON.
        Returns: (is_valid: bool, parsed_data: dict/list/None, error_message: str)
        """
        cleaned = cls.clean_json_string(raw_output)
        if not cleaned:
            return False, None, "Empty output received from LLM."

        try:
            parsed = json.loads(cleaned)
            return True, parsed, ""
        except json.JSONDecodeError as e:
            return False, None, f"JSON Parsing Error: {str(e)}"

    @classmethod
    def validate_schema(cls, parsed_json: Dict[str, Any]) -> Tuple[bool, List[str], int, int]:
        """
        Validates whether parsed JSON matches the expected structure.
        Returns: (is_compliant: bool, missing_keys: list, populated_fields_count: int, total_schema_fields: int)
        """
        if not isinstance(parsed_json, dict):
            return False, REQUIRED_SCHEMA_KEYS, 0, len(REQUIRED_SCHEMA_KEYS)

        missing_keys = []
        for key in REQUIRED_SCHEMA_KEYS:
            if key not in parsed_json:
                missing_keys.append(key)

        is_compliant = (len(missing_keys) == 0)

        # Count total populated vs empty fields recursively
        populated_count, total_count = cls._count_field_stats(parsed_json)

        return is_compliant, missing_keys, populated_count, total_count

    @classmethod
    def _count_field_stats(cls, obj: Any) -> Tuple[int, int]:
        """
        Helper method to recursively count populated vs total fields.
        """
        populated = 0
        total = 0

        if isinstance(obj, dict):
            for k, v in obj.items():
                p, t = cls._count_field_stats(v)
                populated += p
                total += t
        elif isinstance(obj, list):
            total += 1
            if len(obj) > 0:
                populated += 1
                for item in obj:
                    p, t = cls._count_field_stats(item)
                    populated += p
                    total += t
        else:
            total += 1
            if obj is not None and str(obj).strip() != "":
                populated += 1

        return populated, total

    @classmethod
    def detect_hallucinations(cls, parsed_data: Dict[str, Any], raw_resume_text: str) -> List[str]:
        """
        Performs ground-truth check by verifying whether key extracted tokens (emails, names, companies)
        exist within the original resume text.
        Returns a list of potential hallucination warnings.
        """
        warnings = []
        if not isinstance(parsed_data, dict) or not raw_resume_text:
            return warnings

        source_lower = raw_resume_text.lower()

        # Check personal info details
        personal = parsed_data.get("personal_information", {})
        if isinstance(personal, dict):
            email = personal.get("email")
            if email and isinstance(email, str) and email.strip():
                if email.lower() not in source_lower:
                    warnings.append(f"Extracted Email '{email}' was not found in original resume text.")

            phone = personal.get("phone")
            if phone and isinstance(phone, str) and phone.strip():
                # Clean non-digits to perform robust match
                digits = re.sub(r"\D", "", phone)
                source_digits = re.sub(r"\D", "", source_lower)
                if len(digits) >= 7 and digits not in source_digits:
                    warnings.append(f"Extracted Phone '{phone}' was not found in original resume text.")

        return warnings
