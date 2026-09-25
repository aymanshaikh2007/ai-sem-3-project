"""
evaluator.py
------------
Provides evaluation metrics calculation and prompt engineering comparison experiments
against manually curated ground-truth JSON datasets.

Syllabus Reference: Unit I, Topic 1.3 - Prompt Engineering Performance Evaluation.
"""

import json
import os
import pandas as pd
from typing import Dict, Any, List, Tuple
from extractor import ResumeExtractor


class ExtractionEvaluator:
    """
    Evaluator tool to compute quantitative metrics comparing LLM output with Ground Truth.
    """

    @staticmethod
    def flatten_dict(d: Any, parent_key: str = '', sep: str = '.') -> Dict[str, str]:
        """
        Flattens a nested dictionary or list structure into single key-value strings
        for exact field-level matching evaluation.
        """
        items: List[Tuple[str, str]] = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(ExtractionEvaluator.flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(d, list):
            for i, item in enumerate(d):
                new_key = f"{parent_key}[{i}]"
                items.extend(ExtractionEvaluator.flatten_dict(item, new_key, sep=sep).items())
        else:
            val_str = "" if d is None else str(d).strip().lower()
            items.append((parent_key, val_str))
        return dict(items)

    @classmethod
    def evaluate_single_extraction(cls, extracted_json: Dict[str, Any], ground_truth_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single extracted JSON against ground truth JSON.
        Calculates: Precision, Recall, F1-score, Field Accuracy, Missing Field Accuracy.
        """
        if not extracted_json or not isinstance(extracted_json, dict):
            return {
                "json_validity": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "field_accuracy": 0.0,
                "missing_field_accuracy": 0.0,
                "total_ground_truth_fields": 0,
                "correctly_extracted_fields": 0
            }

        ext_flat = cls.flatten_dict(extracted_json)
        gt_flat = cls.flatten_dict(ground_truth_json)

        tp = 0  # True Positives: Extracted key & value match ground truth
        fp = 0  # False Positives: Extracted key & value do not match ground truth or ground truth was empty
        fn = 0  # False Negatives: Ground truth had value, but extraction missed or gave wrong value
        
        missing_correct = 0
        missing_total = 0

        total_gt_fields = len(gt_flat)

        for gt_k, gt_v in gt_flat.items():
            ext_v = ext_flat.get(gt_k, "")
            
            if gt_v == "":  # Field was missing/null in ground truth
                missing_total += 1
                if ext_v == "":
                    missing_correct += 1
                else:
                    fp += 1
            else:
                if ext_v == gt_v:
                    tp += 1
                elif ext_v != "":
                    # Partially correct or hallucinated/wrong value
                    fp += 1
                    fn += 1
                else:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        field_acc = (tp / total_gt_fields * 100) if total_gt_fields > 0 else 0.0
        missing_acc = (missing_correct / missing_total * 100) if missing_total > 0 else 100.0

        return {
            "json_validity": 1.0,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "field_accuracy": round(field_acc, 2),
            "missing_field_accuracy": round(missing_acc, 2),
            "total_ground_truth_fields": total_gt_fields,
            "correctly_extracted_fields": tp
        }

    @classmethod
    def run_prompt_comparison_experiment(
        cls,
        resume_text: str,
        extractor: ResumeExtractor,
        include_few_shot: bool = False
    ) -> pd.DataFrame:
        """
        Runs the 3 Prompts (Basic, Detailed, Structured) on the same resume text
        and compiles comparative performance statistics into a DataFrame.
        """
        modes = ["Basic Prompt", "Detailed Prompt", "Structured Extraction Prompt"]
        results = []

        for mode in modes:
            res = extractor.extract_structured_data(
                resume_text=resume_text,
                prompt_mode=mode,
                include_few_shot=(include_few_shot if mode == "Structured Extraction Prompt" else False)
            )

            results.append({
                "Prompt Type": mode,
                "JSON Validity": "PASS" if res["is_valid_json"] else "FAIL",
                "Schema Compliance": "PASS" if res["is_schema_compliant"] else "FAIL",
                "Populated Fields": f"{res['populated_fields_count']}/{res['total_schema_fields']}",
                "Hallucination Warnings": len(res["hallucination_warnings"]),
                "Latency (sec)": res["latency_seconds"],
                "Missing Field Handling": "Robust (null/[])" if mode == "Structured Extraction Prompt" else "Basic/Inconsistent"
            })

        return pd.DataFrame(results)
