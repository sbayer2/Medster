"""
Vision analysis tool for medical images using Claude's vision API.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json

from medster.model import call_llm


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

        # Call Claude vision API
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
