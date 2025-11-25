# Dynamic code generation tool for custom analysis
# Allows the agent to generate and execute Python code when existing tools are insufficient

from langchain.tools import tool
from typing import Optional
from pydantic import BaseModel, Field
import traceback
import logging
from datetime import datetime

from medster.tools.analysis.primitives import (
    get_patients,
    load_patient,
    search_resources,
    get_conditions,
    get_observations,
    get_medications,
    filter_by_text,
    filter_by_value,
    count_by_field,
    group_by_field,
    aggregate_numeric,
    scan_dicom_directory,
    get_dicom_metadata_from_path,
    find_patient_images,
    load_dicom_image,
    load_ecg_image,
    get_dicom_metadata,
    analyze_image_with_claude,
    analyze_ecg_for_rhythm,
    analyze_multiple_images_with_claude,
    PRIMITIVES_SPEC
)


####################################
# Input Schema
####################################

class CodeGenerationInput(BaseModel):
    analysis_description: str = Field(
        description="Natural language description of the analysis to perform. Be specific about what data to collect and how to aggregate it."
    )
    code: str = Field(
        description=f"Python code to execute using these primitives:\n{PRIMITIVES_SPEC}\nThe code must define a function called 'analyze()' that returns a dict with results."
    )
    patient_limit: int = Field(
        default=50,
        description="Maximum number of patients to analyze (for performance)."
    )


####################################
# Sandbox Environment
####################################

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MEDSTER CODE EXEC - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_sandbox_globals(patient_limit: int) -> dict:
    """Create a restricted global namespace for code execution."""

    def log_progress(message: str):
        """Progress logging function available to generated code."""
        logger.info(message)

    return {
        # FHIR Data Primitives
        "get_patients": lambda limit=patient_limit: get_patients(limit),
        "load_patient": load_patient,
        "search_resources": search_resources,
        "get_conditions": get_conditions,
        "get_observations": get_observations,
        "get_medications": get_medications,
        "filter_by_text": filter_by_text,
        "filter_by_value": filter_by_value,
        "count_by_field": count_by_field,
        "group_by_field": group_by_field,
        "aggregate_numeric": aggregate_numeric,

        # Vision/Imaging Primitives
        "scan_dicom_directory": scan_dicom_directory,
        "get_dicom_metadata_from_path": get_dicom_metadata_from_path,
        "find_patient_images": find_patient_images,
        "load_dicom_image": load_dicom_image,
        "load_ecg_image": load_ecg_image,
        "get_dicom_metadata": get_dicom_metadata,
        "analyze_image_with_claude": analyze_image_with_claude,
        "analyze_ecg_for_rhythm": analyze_ecg_for_rhythm,
        "analyze_multiple_images_with_claude": analyze_multiple_images_with_claude,

        # Safe built-ins
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "any": any,
        "all": all,
        "print": print,
        "hasattr": hasattr,
        "getattr": getattr,
        "isinstance": isinstance,
        "type": type,
        "ord": ord,  # For hash-based pseudo-random selection
        "chr": chr,  # Inverse of ord, useful for string operations

        # Exception handling
        "Exception": Exception,

        # Progress logging
        "log_progress": log_progress,

        # No dangerous functions
        "__builtins__": {},
    }


####################################
# Tool
####################################

@tool(args_schema=CodeGenerationInput)
def generate_and_run_analysis(
    analysis_description: str,
    code: str,
    patient_limit: int = 50
) -> dict:
    """
    Generates and executes custom Python analysis code using FHIR and vision primitives.
    Use this when existing tools don't support the specific analysis pattern needed.

    The code MUST define a function called 'analyze()' that returns a dict.

    Example code structure:
    ```
    def analyze():
        patients = get_patients(50)
        results = []
        for pid in patients:
            bundle = load_patient(pid)
            conditions = get_conditions(bundle)
            # ... custom analysis logic ...
        return {{"summary": results, "count": len(results)}}
    ```

    Example with autonomous vision analysis:
    ```
    def analyze():
        patients = get_patients(10)
        results = []
        for pid in patients:
            # Check for arrhythmia diagnosis
            bundle = load_patient(pid)
            conditions = get_conditions(bundle)
            has_arrhythmia = any('arrhythmia' in c.get('name', '').lower() for c in conditions)

            if has_arrhythmia:
                # Load and analyze ECG in one step
                ecg = load_ecg_image(pid)
                if ecg:
                    afib_analysis = analyze_image_with_claude(
                        ecg,
                        f"Analyze this ECG for patient {{pid}}. Does it show atrial fibrillation pattern? Answer yes or no with key findings."
                    )
                    results.append({{"patient": pid, "afib_analysis": afib_analysis}})
        return {{"afib_results": results}}
    ```

    Available primitives:
    - FHIR: get_patients, load_patient, get_conditions, get_observations, get_medications
    - Filtering: filter_by_text, filter_by_value
    - Aggregation: count_by_field, group_by_field, aggregate_numeric
    - Vision Loading: find_patient_images, load_dicom_image, load_ecg_image, get_dicom_metadata
    - Vision Analysis: analyze_image_with_claude, analyze_multiple_images_with_claude

    NOTE: You can now perform complete autonomous vision analysis within generated code!
    Use analyze_image_with_claude() to analyze images directly in your code without
    needing a separate tool call.
    """
    try:
        logger.info(f"Starting code execution: {analysis_description}")
        logger.info(f"Patient limit: {patient_limit}")

        # Create restricted sandbox
        sandbox_globals = create_sandbox_globals(patient_limit)
        sandbox_locals = {}

        # Execute the generated code
        exec(code, sandbox_globals, sandbox_locals)

        # Check if analyze function was defined
        if "analyze" not in sandbox_locals:
            return {
                "status": "error",
                "error": "Code must define a function called 'analyze()'",
                "description": analysis_description
            }

        # Run the analysis
        logger.info("Executing analyze() function...")
        start_time = datetime.now()
        result = sandbox_locals["analyze"]()
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {elapsed:.2f} seconds")

        return {
            "status": "success",
            "description": analysis_description,
            "patient_limit": patient_limit,
            "result": result
        }

    except SyntaxError as e:
        return {
            "status": "error",
            "error": f"Syntax error in generated code: {str(e)}",
            "line": e.lineno,
            "description": analysis_description
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Execution error: {str(e)}",
            "traceback": traceback.format_exc(),
            "description": analysis_description
        }


# Export the primitives spec for the agent to reference
def get_primitives_spec() -> str:
    """Return the API specification for code generation."""
    return PRIMITIVES_SPEC
