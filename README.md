# 📄 AI-Powered Resume Information Extraction System

An end-to-end AI-powered system that parses unstructured resume documents (PDF format or plain text) and extracts candidate details into a strictly formatted, schema-compliant JSON structure using **Prompt Engineering** techniques and automated validation.

---

## 🌟 Key Features

- 📄 **Single Resume Extraction**: Upload a PDF or paste raw resume text to extract personal info, education, categorized technical skills, work experience, projects, and certifications.
- 🧪 **Prompt Engineering Experimentation Engine**: Real-time side-by-side comparison of three distinct prompt strategies:
  1. **Basic Prompt:** Simple, unstructured prompt (demonstrates syntax errors / unquoted JSON keys).
  2. **Detailed Prompt:** Detailed instructions without strict target schema (demonstrates schema mismatch).
  3. **Structured Extraction Prompt:** Strict role, explicit JSON target schema, grounding rules, and null handling (100% schema compliant).
- 📊 **Batch Benchmark & Evaluation Engine**: Evaluates extraction quality against ground-truth datasets, calculating **Precision**, **Recall**, **F1-Score**, and **Field Accuracy**.
- ⚡ **Automated Offline Local Engine**: Built-in high-performance heuristic extraction simulator enabling zero-cost, zero-latency, offline testing and presentation without external API requirements.
- 📥 **Data Export Options**: Download extracted structured data as formatted `.json` or flattened `.csv` files.

---

## 📊 System Performance & Benchmark Results

Evaluated against test cases in the `test_data/` directory:

| Metric | Score |
| :--- | :--- |
| **Average Precision** | **92.76%** |
| **Average Recall** | **88.54%** |
| **Average F1-Score** | **90.58%** |
| **Field Accuracy** | **76.80%** |

---

## 🏗️ System Workflow Architecture

```text
+-------------------------------------------------+
|               User Resume Input                 |
|       (PDF Upload or Plain Text Paste)          |
+-------------------------------------------------+
                        |
                        v
+-------------------------------------------------+
|             PDF Text Extraction                 |
|             (pypdf / regex cleaning)            |
+-------------------------------------------------+
                        |
                        v
+-------------------------------------------------+
|            Prompt Engineering Engine            |
| (Role + Task + JSON Schema + Grounding Rules)   |
+-------------------------------------------------+
                        |
                        v
+-------------------------------------------------+
|          Automated Extraction Engine            |
|       (LLM API / High-Fidelity Simulator)       |
+-------------------------------------------------+
                        |
                        v
+-------------------------------------------------+
|             JSON Validation & Clean             |
|   (Syntax Parsing + Schema Match + Hallucination|
+-------------------------------------------------+
                        |
                        v
+-------------------------------------------------+
|          Structured Output & Export             |
|       (JSON Viewer, Table, CSV Download)        |
+-------------------------------------------------+
```

---

## 📁 Repository Structure

```text
ai-sem-3-project/
├── app.py                  # Streamlit Web Application entrypoint
├── extractor.py            # PDF text extraction and LLM/Simulator interface
├── prompt_engineering.py   # Prompt engineering templates (Basic, Detailed, Structured)
├── validator.py            # JSON syntax cleaner, schema validator, and hallucination detector
├── evaluator.py            # Evaluation metrics engine (Precision, Recall, F1, Accuracy)
├── test_data/              # Ground truth JSON datasets and test resume text files
├── prompts/                # Prompt text templates
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## 📋 Target JSON Schema

```json
{
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
  "certifications": [],
  "other_information": {
    "achievements": [],
    "languages_spoken": [],
    "areas_of_interest": []
  }
}
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/aymanshaikh2007/ai-sem-3-project.git
cd ai-sem-3-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`.
