"""
Vision analysis tool for medical images using Claude's vision API.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

from medster.model import call_llm
from medster.tools.analysis.primitives import (
    load_ecg_image,
    load_dicom_image,
    find_patient_images,
    analyze_ecg_for_rhythm
)


class PatientECGAnalysisInput(BaseModel):
    """Input schema for patient ECG analysis."""

    patient_id: str = Field(
        description="Patient UUID to analyze ECG for"
    )
    clinical_question: str = Field(
        default="Analyze the ECG tracing and describe the cardiac rhythm including rate, regularity, P waves, QRS complexes, and any abnormalities",
        description="Specific clinical question about the ECG"
    )
    clinical_context: str = Field(
        default="",
        description="Optional clinical context (e.g., 'Patient with hypertension and diabetes')"
    )


@tool(args_schema=PatientECGAnalysisInput)
def analyze_patient_ecg(
    patient_id: str,
    clinical_question: str = "Analyze the ECG tracing and describe the cardiac rhythm including rate, regularity, P waves, QRS complexes, and any abnormalities",
    clinical_context: str = ""
) -> dict:
    """
    Analyze a patient's ECG image using Claude's vision API.

    This tool takes a patient_id and automatically loads their ECG image,
    then performs vision analysis to answer clinical questions about the ECG.

    Use this when you have a patient ID and want to visually analyze their ECG tracing.
    The tool handles image loading internally - you don't need base64 data.

    Returns structured analysis including rhythm classification, findings, and clinical significance.
    """
    try:
        # Use the structured ECG analysis primitive
        result = analyze_ecg_for_rhythm(patient_id, clinical_context)

        if not result.get("ecg_available", False):
            return {
                "status": "error",
                "patient_id": patient_id,
                "error": "No ECG image available for this patient"
            }

        # If custom question, do additional analysis
        if "atrial fibrillation" not in clinical_question.lower() and "rhythm" not in clinical_question.lower():
            # Load image for custom analysis
            ecg_image = load_ecg_image(patient_id)
            if ecg_image:
                context_str = f" (Clinical context: {clinical_context})" if clinical_context else ""
                prompt = f"""Analyze this ECG tracing for patient {patient_id}{context_str}.

{clinical_question}

Provide a detailed analysis with specific findings."""

                response = call_llm(
                    prompt=prompt,
                    images=[ecg_image],
                    model="claude-sonnet-4.5"  # Use Sonnet 4.5 for vision analysis
                )
                custom_analysis = response.content if hasattr(response, 'content') else str(response)
                result["custom_analysis"] = custom_analysis

        return {
            "status": "success",
            "patient_id": patient_id,
            "ecg_available": True,
            "rhythm": result.get("rhythm", "Unknown"),
            "afib_detected": result.get("afib_detected", False),
            "rr_intervals": result.get("rr_intervals", "Unknown"),
            "p_waves": result.get("p_waves", "Unknown"),
            "baseline": result.get("baseline", "Unknown"),
            "confidence": result.get("confidence", "Unknown"),
            "clinical_significance": result.get("clinical_significance", ""),
            "clinical_context": clinical_context,
            "detailed_analysis": result.get("raw_analysis", "")
        }

    except Exception as e:
        return {
            "status": "error",
            "patient_id": patient_id,
            "error": f"ECG analysis failed: {str(e)}"
        }


class VisionAnalysisInput(BaseModel):
    """Input schema for vision analysis."""

    analysis_prompt: str = Field(
        description="Specific clinical question to answer about the images (e.g., 'Does this ECG show atrial fibrillation pattern?', 'Identify any masses or hemorrhage in this brain MRI')"
    )
    image_data: List[Dict[str, Any]] = Field(
        description="List of image objects, each containing 'image_base64' (required), 'patient_id' (optional), 'modality' (optional), 'context' (optional)"
    )
    max_images: int = Field(
        default=3,
        description="Maximum number of images to analyze in a single call (for token efficiency)"
    )


@tool(args_schema=VisionAnalysisInput)
def analyze_medical_images(
    analysis_prompt: str,
    image_data: List[Dict[str, Any]],
    max_images: int = 3
) -> dict:
    """
    Analyze medical images using Claude's vision API.

    Use this tool when you have loaded base64-encoded images (DICOM, ECG, etc.)
    and need to analyze them for clinical findings.

    The tool accepts a list of image objects with:
    - image_base64 (required): Base64-encoded PNG image string
    - patient_id (optional): Patient identifier
    - modality (optional): Imaging modality (MRI, CT, ECG, etc.)
    - context (optional): Clinical context for the image

    Example usage after generate_and_run_analysis loads images:
    - analysis_prompt: "Analyze these ECG waveforms for atrial fibrillation pattern"
    - image_data: List of dicts with patient_id, image_base64, and modality fields

    Returns a structured analysis with findings for each image.
    """
    try:
        # Limit images for token efficiency
        images_to_analyze = image_data[:max_images]

        # Extract base64 images
        base64_images = []
        patient_context = []

        for idx, img in enumerate(images_to_analyze):
            if "image_base64" not in img:
                continue

            base64_images.append(img["image_base64"])

            # Build context for each image
            context_parts = [f"Image {idx + 1}"]
            if "patient_id" in img:
                context_parts.append(f"Patient: {img['patient_id']}")
            if "modality" in img:
                context_parts.append(f"Modality: {img['modality']}")
            if "context" in img:
                context_parts.append(img["context"])

            patient_context.append(" | ".join(context_parts))

        if not base64_images:
            return {
                "status": "error",
                "error": "No valid images found in image_data (missing 'image_base64' key)"
            }

        # Build prompt with context
        full_prompt = f"""You are analyzing medical images for clinical decision support.

{analysis_prompt}

Context for each image:
{chr(10).join(f"- {ctx}" for ctx in patient_context)}

For each image, provide:
1. Patient ID (if provided)
2. Key visual findings
3. Direct answer to the clinical question
4. Any critical findings requiring immediate attention

Format your response as structured findings for each image."""

        # Call Claude vision API with Sonnet 4.5 for vision analysis
        response = call_llm(
            prompt=full_prompt,
            images=base64_images,
            model="claude-sonnet-4.5"
        )

        # Extract text content from response
        analysis_text = response.content if hasattr(response, 'content') else str(response)

        return {
            "status": "success",
            "images_analyzed": len(base64_images),
            "clinical_question": analysis_prompt,
            "vision_analysis": analysis_text,
            "patient_contexts": patient_context
        }

    except Exception as e:
        return {
            "status": "error",
            "error": f"Vision analysis failed: {str(e)}",
            "images_attempted": len(image_data)
        }
