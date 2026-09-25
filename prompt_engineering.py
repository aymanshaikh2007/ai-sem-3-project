"""
prompt_engineering.py
---------------------
Manages the prompt templates and construction logic for extracting structured
information from resumes using different prompt engineering techniques.

Syllabus Reference: Unit I, Topic 1.3 - Introduction to Prompt Engineering and Modern AI Tooling.
"""

import json
from typing import Dict, Any, Optional, List

# Target JSON Schema description for prompts
RESUME_JSON_SCHEMA = {
    "personal_information": {
        "full_name": "String or null",
        "email": "String or null",
        "phone": "String or null",
        "location": "String or null",
        "linkedin": "String or null",
        "github": "String or null",
        "portfolio": "String or null"
    },
    "education": [
        {
            "degree": "String or null",
            "field_of_study": "String or null",
            "institution": "String or null",
            "graduation_year": "String or null",
            "grade_cgpa": "String or null"
        }
    ],
    "skills": {
        "programming_languages": ["String"],
        "frameworks_and_libraries": ["String"],
        "databases": ["String"],
        "tools_and_technologies": ["String"],
        "other_technical_skills": ["String"]
    },
    "work_experience": [
        {
            "company": "String or null",
            "job_title": "String or null",
            "duration": "String or null",
            "responsibilities": ["String"]
        }
    ],
    "projects": [
        {
            "project_name": "String or null",
            "description": "String or null",
            "technologies_used": ["String"]
        }
    ],
    "certifications": [
        {
            "certification_name": "String or null",
            "issuing_organization": "String or null",
            "year": "String or null"
        }
    ],
    "other_information": {
        "achievements": ["String"],
        "languages_spoken": ["String"],
        "areas_of_interest": ["String"]
    }
}

# Few-shot example definition
FEW_SHOT_EXAMPLE_INPUT = """
John Doe
Email: john.doe@email.com | Phone: +91 9876543210 | Location: Mumbai, India
LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

EDUCATION:
B.Sc. in Artificial Intelligence, University of Mumbai (2021 - 2024), CGPA: 8.9/10

SKILLS:
- Languages: Python, C++, SQL
- Frameworks: Streamlit, PyTorch
- Tools: Git, VS Code

EXPERIENCE:
AI Intern at Tech Corp (June 2023 - Aug 2023)
- Built a text classification script using Python.
"""

FEW_SHOT_EXAMPLE_OUTPUT = {
    "personal_information": {
        "full_name": "John Doe",
        "email": "john.doe@email.com",
        "phone": "+91 9876543210",
        "location": "Mumbai, India",
        "linkedin": "linkedin.com/in/johndoe",
        "github": "github.com/johndoe",
        "portfolio": None
    },
    "education": [
        {
            "degree": "B.Sc.",
            "field_of_study": "Artificial Intelligence",
            "institution": "University of Mumbai",
            "graduation_year": "2024",
            "grade_cgpa": "8.9/10"
        }
    ],
    "skills": {
        "programming_languages": ["Python", "C++", "SQL"],
        "frameworks_and_libraries": ["Streamlit", "PyTorch"],
        "databases": [],
        "tools_and_technologies": ["Git", "VS Code"],
        "other_technical_skills": []
    },
    "work_experience": [
        {
            "company": "Tech Corp",
            "job_title": "AI Intern",
            "duration": "June 2023 - Aug 2023",
            "responsibilities": ["Built a text classification script using Python."]
        }
    ],
    "projects": [],
    "certifications": [],
    "other_information": {
        "achievements": [],
        "languages_spoken": [],
        "areas_of_interest": []
    }
}

class PromptEngine:
    """
    Manages generation of Basic, Detailed, and Structured prompts,
    supporting Zero-Shot and Few-Shot configurations.
    """

    @staticmethod
    def get_basic_prompt(resume_text: str) -> str:
        """
        Prompt 1: Basic / Naive Prompt.
        Simple request without explicit rules or JSON schema enforcing.
        """
        return f"""Extract the details from this resume and convert it into JSON format:

RESUME TEXT:
{resume_text}
"""

    @staticmethod
    def get_detailed_prompt(resume_text: str) -> str:
        """
        Prompt 2: Detailed Instruction-Based Prompt.
        Provides field listings and key instructions, but lacks strict JSON schema specification.
        """
        return f"""You are a professional HR assistant. Please analyze the following resume and extract all relevant candidate information into a JSON object.

Extract these sections:
- Personal Info (Name, Email, Phone, Location, LinkedIn, GitHub)
- Education (Degree, Branch, College, Year, GPA)
- Technical Skills (Programming, Frameworks, Tools)
- Experience (Company, Role, Dates, Responsibilities)
- Projects (Title, Description, Tech Stack)
- Certifications
- Languages and Interests

Rules:
1. Output valid JSON only.
2. If any detail is missing, set it to null or empty list.
3. Do not make up facts.

RESUME TEXT:
{resume_text}
"""

    @staticmethod
    def get_structured_prompt(resume_text: str, include_few_shot: bool = False) -> str:
        """
        Prompt 3: Structured Extraction Prompt.
        Includes explicit Role, Task, Target JSON Schema, Rules, Missing-value handling,
        Formatting constraints, and optional Few-Shot guidance.
        """
        schema_str = json.dumps(RESUME_JSON_SCHEMA, indent=2)
        
        few_shot_block = ""
        if include_few_shot:
            few_shot_block = f"""
=================== EXAMPLE FEW-SHOT INSTRUCTION ===================
EXAMPLE INPUT:
{FEW_SHOT_EXAMPLE_INPUT}

EXAMPLE EXPECTED JSON OUTPUT:
{json.dumps(FEW_SHOT_EXAMPLE_OUTPUT, indent=2)}
===================================================================
"""

        return f"""You are an Expert AI Resume Parser specializing in structured information extraction.

### TASK:
Analyze the unstructured resume text provided below and extract all candidate data into a strictly valid JSON object matching the EXACT target schema provided.

### TARGET JSON SCHEMA:
{schema_str}
{few_shot_block}
### EXTRACTION & GROUNDING RULES:
1. STRICT TRUTH / NO HALLUCINATION: Extract details ONLY if explicitly present in the text. NEVER guess, assume, or fabricate any data.
2. MISSING VALUES: If a field is missing or not mentioned in the resume:
   - For string fields, set the value strictly to `null`.
   - For array/list fields, set the value strictly to an empty list `[]`.
3. CLEAN FORMATTING:
   - Clean up spacing and noise.
   - Convert email to lowercase.
   - Separate skill items into distinct string elements inside array fields.
4. STRICTV ALID JSON ONLY:
   - Your response MUST contain ONLY raw valid JSON.
   - Do NOT wrap your output in markdown code blocks like ```json ... ```.
   - Do NOT add any intro, outro, explanations, or commentary.

### INPUT RESUME TEXT:
{resume_text}
"""

    @classmethod
    def get_prompt_by_mode(cls, mode: str, resume_text: str, include_few_shot: bool = False) -> str:
        """
        Returns the built prompt string based on the requested extraction mode.
        """
        if mode == "Basic Prompt":
            return cls.get_basic_prompt(resume_text)
        elif mode == "Detailed Prompt":
            return cls.get_detailed_prompt(resume_text)
        elif mode == "Structured Extraction Prompt":
            return cls.get_structured_prompt(resume_text, include_few_shot=include_few_shot)
        else:
            return cls.get_structured_prompt(resume_text, include_few_shot=include_few_shot)
