"""
extractor.py
------------
Handles text extraction from PDF files and interacts with the OpenAI LLM API
(with an offline fallback simulator for offline presentation/testing).
"""

import os
import time
import json
import re
from typing import Dict, Any, Tuple, Optional
import pypdf

from prompt_engineering import PromptEngine, FEW_SHOT_EXAMPLE_OUTPUT
from validator import OutputValidator

# Try importing OpenAI library
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ResumeExtractor:
    """
    Main extraction engine for parsing PDFs and querying the LLM model.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.client = None

        if OPENAI_AVAILABLE and self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"):
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    @staticmethod
    def extract_text_from_pdf(file_stream) -> str:
        """
        Extracts plain text content from a PDF file stream using pypdf.
        """
        try:
            reader = pypdf.PdfReader(file_stream)
            extracted_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
            return "\n\n".join(extracted_text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def extract_structured_data(
        self,
        resume_text: str,
        prompt_mode: str = "Structured Extraction Prompt",
        include_few_shot: bool = False,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Sends the resume text to the LLM (or mock simulator if API key is not configured)
        and returns structured result along with evaluation metadata.
        """
        prompt = PromptEngine.get_prompt_by_mode(
            mode=prompt_mode,
            resume_text=resume_text,
            include_few_shot=include_few_shot
        )

        start_time = time.time()

        if self.client:
            raw_response, status_code, err_msg = self._call_openai_api(prompt, temperature)
            is_mock = False
        else:
            raw_response = self._simulate_offline_llm(resume_text, prompt_mode)
            status_code = 200
            err_msg = ""
            is_mock = True

        latency_seconds = round(time.time() - start_time, 3)

        # Validate JSON Syntax
        is_valid_json, parsed_data, json_err = OutputValidator.validate_json_syntax(raw_response)

        # Validate Schema Compliance & Stats
        if is_valid_json and isinstance(parsed_data, dict):
            is_schema_compliant, missing_keys, populated_count, total_count = OutputValidator.validate_schema(parsed_data)
            hallucinations = OutputValidator.detect_hallucinations(parsed_data, resume_text)
        else:
            is_schema_compliant = False
            missing_keys = ["all"]
            populated_count = 0
            total_count = 0
            hallucinations = []

        return {
            "prompt_mode": prompt_mode,
            "include_few_shot": include_few_shot,
            "constructed_prompt": prompt,
            "raw_llm_response": raw_response,
            "is_valid_json": is_valid_json,
            "json_error": json_err,
            "parsed_json": parsed_data if is_valid_json else None,
            "is_schema_compliant": is_schema_compliant,
            "missing_schema_keys": missing_keys,
            "populated_fields_count": populated_count,
            "total_schema_fields": total_count,
            "hallucination_warnings": hallucinations,
            "latency_seconds": latency_seconds,
            "is_offline_mock": is_mock,
            "error_message": err_msg
        }

    def _call_openai_api(self, prompt: str, temperature: float) -> Tuple[str, int, str]:
        """
        Internal execution wrapper for OpenAI API call.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a specialized JSON data extraction engine."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content, 200, ""
        except Exception as e:
            return f"API Error: {str(e)}", 500, str(e)

    def _simulate_offline_llm(self, resume_text: str, prompt_mode: str) -> str:
        """
        Offline fallback LLM simulator based on heuristic regex parsing.
        Ensures the college project can be demonstrated even without an active OpenAI API Key!
        """
        time.sleep(0.5) # Simulate network latency

        if prompt_mode == "Basic Prompt":
            # Basic prompt sometimes produces flawed JSON or missing quotes in naive mode
            return f"""Here is the extracted resume JSON:
{{
    name: "{self._extract_regex(resume_text, r'([A-Z][a-z]+\s+[A-Z][a-z]+)', 'Candidate Name')}",
    email: "{self._extract_regex(resume_text, r'[\w\.-]+@[\w\.-]+\.\w+', 'null')}",
    phone: "{self._extract_regex(resume_text, r'[\+\d\s\-\(\)]{10,15}', 'null')}",
    skills: ["Python", "Machine Learning"]
}}"""

        # For Detailed and Structured Prompts, generate proper schema compliant JSON
        email = self._extract_regex(resume_text, r'[\w\.-]+@[\w\.-]+\.\w+', None)
        phone = self._extract_regex(resume_text, r'(\+?\d[\d\s\-]{8,14}\d)', None)
        linkedin = self._extract_regex(resume_text, r'(linkedin\.com/in/[\w\-]+)', None)
        github = self._extract_regex(resume_text, r'(github\.com/[\w\-]+)', None)

        # Extract name from first line or pattern
        lines = [l.strip() for l in resume_text.strip().split('\n') if l.strip()]
        full_name = lines[0] if lines else "Candidate Name"
        if len(full_name) > 40 or "@" in full_name:
            full_name = "Candidate Name"

        # Heuristic skill extraction
        known_skills = ["Python", "Java", "C++", "SQL", "HTML", "CSS", "JavaScript", "React", "Node", "Pandas", "NumPy", "PyTorch", "TensorFlow", "Streamlit", "Git", "Docker", "AWS"]
        extracted_skills = [s for s in known_skills if re.search(r'\b' + re.escape(s) + r'\b', resume_text, re.IGNORECASE)]

        result = {
            "personal_information": {
                "full_name": full_name,
                "email": email,
                "phone": phone,
                "location": "Mumbai, Maharashtra" if "mumbai" in resume_text.lower() else None,
                "linkedin": linkedin,
                "github": github,
                "portfolio": None
            },
            "education": [
                {
                    "degree": "B.Sc." if "b.sc" in resume_text.lower() else "Bachelor of Engineering",
                    "field_of_study": "Computer Science" if "computer science" in resume_text.lower() else "Artificial Intelligence",
                    "institution": "Mumbai University",
                    "graduation_year": "2024",
                    "grade_cgpa": "8.5/10"
                }
            ],
            "skills": {
                "programming_languages": [s for s in extracted_skills if s in ["Python", "Java", "C++", "JavaScript", "SQL"]],
                "frameworks_and_libraries": [s for s in extracted_skills if s in ["Streamlit", "React", "Node", "Pandas", "NumPy", "PyTorch", "TensorFlow"]],
                "databases": ["MySQL"] if "sql" in resume_text.lower() else [],
                "tools_and_technologies": [s for s in extracted_skills if s in ["Git", "Docker", "AWS"]],
                "other_technical_skills": ["Machine Learning"] if "machine learning" in resume_text.lower() else []
            },
            "work_experience": [
                {
                    "company": "Tech Solutions",
                    "job_title": "Python Developer",
                    "duration": "2022 - 2024",
                    "responsibilities": ["Developed backend API endpoints.", "Managed database queries."]
                }
            ] if "experience" in resume_text.lower() or "developer" in resume_text.lower() else [],
            "projects": [
                {
                    "project_name": "Resume Information Extraction System",
                    "description": "Built AI powered resume parser using Prompt Engineering.",
                    "technologies_used": ["Python", "Streamlit", "OpenAI"]
                }
            ] if "project" in resume_text.lower() else [],
            "certifications": [],
            "other_information": {
                "achievements": [],
                "languages_spoken": ["English", "Hindi"],
                "areas_of_interest": ["Artificial Intelligence", "NLP"]
            }
        }
        return json.dumps(result, indent=2)

    def _extract_regex(self, text: str, pattern: str, default: Any = None) -> Any:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(0).strip() if match else default
