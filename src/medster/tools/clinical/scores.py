from langchain.tools import tool
from typing import Literal, Optional
from pydantic import BaseModel, Field

####################################
# Input Schemas
####################################

class ClinicalScoreInput(BaseModel):
    score_type: Literal[
        "wells_dvt", "wells_pe", "chadsvasc", "hasbled",
        "apache_ii", "sofa", "curb65", "meld", "child_pugh"
    ] = Field(description="The clinical scoring system to calculate.")
    parameters: dict = Field(description="Parameters required for the score calculation. Each score has specific required fields.")


####################################
# Score Calculation Functions
####################################

def calculate_wells_dvt(params: dict) -> dict:
    """Calculate Wells' Criteria for DVT probability."""
    score = 0

    # Active cancer
    if params.get("active_cancer", False):
        score += 1

    # Paralysis, paresis, or recent plaster immobilization
    if params.get("paralysis_or_immobilization", False):
        score += 1

    # Recently bedridden >3 days or major surgery within 12 weeks
    if params.get("bedridden_or_surgery", False):
        score += 1

    # Localized tenderness along deep venous system
    if params.get("localized_tenderness", False):
        score += 1

    # Entire leg swelling
    if params.get("leg_swelling", False):
        score += 1

    # Calf swelling >3cm compared to asymptomatic leg
    if params.get("calf_swelling_3cm", False):
        score += 1

    # Pitting edema
    if params.get("pitting_edema", False):
        score += 1

    # Collateral superficial veins
    if params.get("collateral_veins", False):
        score += 1

    # Previously documented DVT
    if params.get("previous_dvt", False):
        score += 1

    # Alternative diagnosis at least as likely as DVT
    if params.get("alternative_diagnosis", False):
        score -= 2

    # Interpretation
    if score <= 0:
        risk = "Low"
        probability = "5%"
    elif score <= 2:
        risk = "Moderate"
        probability = "17%"
    else:
        risk = "High"
        probability = "53%"

    return {
        "score_name": "Wells' Criteria for DVT",
        "score": score,
        "risk_category": risk,
        "dvt_probability": probability,
        "recommendation": f"{risk} probability - consider D-dimer and/or ultrasound based on clinical judgment"
    }


def calculate_chadsvasc(params: dict) -> dict:
    """Calculate CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk."""
    score = 0

    # C - Congestive heart failure
    if params.get("chf", False):
        score += 1

    # H - Hypertension
    if params.get("hypertension", False):
        score += 1

    # A2 - Age >= 75
    if params.get("age_75_or_older", False):
        score += 2
    elif params.get("age_65_to_74", False):
        score += 1

    # D - Diabetes mellitus
    if params.get("diabetes", False):
        score += 1

    # S2 - Stroke/TIA/thromboembolism
    if params.get("stroke_tia", False):
        score += 2

    # V - Vascular disease
    if params.get("vascular_disease", False):
        score += 1

    # Sc - Sex category (female)
    if params.get("female", False):
        score += 1

    # Risk interpretation
    if score == 0:
        risk = "Low"
        recommendation = "No anticoagulation recommended"
    elif score == 1:
        risk = "Low-Moderate"
        recommendation = "Consider anticoagulation"
    else:
        risk = "Moderate-High"
        recommendation = "Anticoagulation recommended"

    return {
        "score_name": "CHA2DS2-VASc Score",
        "score": score,
        "risk_category": risk,
        "recommendation": recommendation
    }


def calculate_curb65(params: dict) -> dict:
    """Calculate CURB-65 Score for Pneumonia Severity."""
    score = 0

    # C - Confusion (new)
    if params.get("confusion", False):
        score += 1

    # U - Urea > 7 mmol/L (BUN > 19 mg/dL)
    if params.get("urea_elevated", False):
        score += 1

    # R - Respiratory rate >= 30
    if params.get("respiratory_rate_30", False):
        score += 1

    # B - Blood pressure (SBP < 90 or DBP <= 60)
    if params.get("low_blood_pressure", False):
        score += 1

    # 65 - Age >= 65
    if params.get("age_65_or_older", False):
        score += 1

    # Risk interpretation
    if score <= 1:
        risk = "Low"
        mortality = "1.5%"
        recommendation = "Consider outpatient treatment"
    elif score == 2:
        risk = "Moderate"
        mortality = "9.2%"
        recommendation = "Consider short inpatient stay or closely supervised outpatient"
    else:
        risk = "High"
        mortality = "22%"
        recommendation = "Hospitalize, consider ICU if score 4-5"

    return {
        "score_name": "CURB-65 Pneumonia Severity",
        "score": score,
        "risk_category": risk,
        "30_day_mortality": mortality,
        "recommendation": recommendation
    }


def calculate_meld(params: dict) -> dict:
    """Calculate MELD Score for End-Stage Liver Disease."""
    import math

    # Get values with defaults
    creatinine = max(1.0, min(4.0, params.get("creatinine", 1.0)))
    bilirubin = max(1.0, params.get("bilirubin", 1.0))
    inr = max(1.0, params.get("inr", 1.0))
    dialysis = params.get("dialysis", False)

    # If on dialysis, set creatinine to 4
    if dialysis:
        creatinine = 4.0

    # MELD formula
    meld_score = (
        0.957 * math.log(creatinine) +
        0.378 * math.log(bilirubin) +
        1.120 * math.log(inr) +
        0.643
    ) * 10

    meld_score = round(meld_score)
    meld_score = max(6, min(40, meld_score))

    # Mortality interpretation
    if meld_score < 10:
        mortality_3month = "1.9%"
    elif meld_score < 20:
        mortality_3month = "6.0%"
    elif meld_score < 30:
        mortality_3month = "19.6%"
    elif meld_score < 40:
        mortality_3month = "52.6%"
    else:
        mortality_3month = "71.3%"

    return {
        "score_name": "MELD Score",
        "score": meld_score,
        "3_month_mortality": mortality_3month,
        "note": "Higher scores indicate more urgent need for transplant"
    }


####################################
# Main Tool
####################################

@tool(args_schema=ClinicalScoreInput)
def calculate_clinical_score(
    score_type: str,
    parameters: dict
) -> dict:
    """
    Calculates clinical risk scores including Wells' Criteria, CHA2DS2-VASc,
    CURB-65, MELD, and others. Provides risk stratification and recommendations.
    IMPORTANT: These are decision support tools - always use clinical judgment.
    """
    calculators = {
        "wells_dvt": calculate_wells_dvt,
        "chadsvasc": calculate_chadsvasc,
        "curb65": calculate_curb65,
        "meld": calculate_meld,
    }

    if score_type not in calculators:
        return {
            "error": f"Score type '{score_type}' not implemented",
            "available_scores": list(calculators.keys())
        }

    try:
        result = calculators[score_type](parameters)
        result["disclaimer"] = "Clinical scores are decision support tools. Always use clinical judgment."
        return result
    except Exception as e:
        return {
            "score_type": score_type,
            "error": str(e)
        }
