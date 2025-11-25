# Medster - Autonomous Clinical Case Analysis Agent

An autonomous agent for deep clinical case analysis, inspired by [Dexter](https://github.com/virattt/dexter) and adapted for medical domain.

## Overview

Medster "thinks, plans, and learns as it works" - performing clinical analysis through task planning, self-reflection, and real-time medical data. It leverages SYNTHEA/FHIR data sources and integrates with your MCP medical analysis server for sophisticated clinical reasoning.

## Core Capabilities

- **Intelligent Task Planning**: Breaks down complex clinical questions into structured diagnostic and therapeutic steps
- **Autonomous Execution**: Automatically selects and runs appropriate tools for data gathering (labs, notes, vitals, imaging, medications)
- **Self-Validation**: Verifies its own work and iterates until tasks are complete
- **Real-Time Medical Data**: Patient notes, lab results, vital sign trends, medication lists
- **Safety Mechanisms**: Loop detection, critical value flagging, drug interaction checking

## Primary Use Cases

1. **Clinical Case Analysis** - Comprehensive review of patient cases with risk stratification
2. **Differential Diagnosis Workup** - Prioritized differentials with optimal diagnostic sequences

## Architecture

```
                    MEDSTER CLI
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Coherent │  │   MCP    │  │ Claude   │
    │ Data Set │  │  Server  │  │ Sonnet   │
    │          │  │          │  │   4.5    │
    │ FHIR     │  │ Analyze  │  │ Planning │
    │ DICOM    │  │ Complex  │  │ Reasoning│
    │ ECG/Notes│  │ Notes    │  │ Synthesis│
    └──────────┘  └──────────┘  └──────────┘
```

## Requirements

- Python 3.10+
- Anthropic API key (for Claude Sonnet 4.5)
- FHIR server with patient data (default: HAPI FHIR test server)
- Optional: Your MCP medical analysis server for complex note analysis

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Medster
```

2. Install dependencies using uv:
```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

3. Configure environment:
```bash
cp env.example .env
```

4. Get your Anthropic API key:
   - Sign up at https://console.anthropic.com/
   - Go to API Keys section
   - Create a new API key
   - Copy the key (starts with `sk-ant-`)

5. Edit `.env` file with your credentials:
```bash
# Required: Your Anthropic API key for Claude Sonnet 4.5
ANTHROPIC_API_KEY=sk-ant-your_actual_key_here

# Required: Path to Coherent Data Set FHIR folder
COHERENT_DATA_PATH=./coherent_data/fhir

# Optional: Your MCP medical analysis server
MCP_SERVER_URL=http://localhost:8000
MCP_API_KEY=your_mcp_key_if_needed
```

**⚠️ Security Note:** Never commit your `.env` file to git. It's already in `.gitignore` to protect your API keys.

## Usage

Run the interactive CLI:
```bash
uv run medster-agent
```

Or:
```bash
python -m medster.cli
```

### Example Queries

**Clinical Case Analysis:**
```
medster>> Analyze this patient - 58yo male with chest pain, elevated troponins, and new ECG changes. What's the diagnostic workup and risk stratification?
```

**Lab Review:**
```
medster>> Get the last 7 days of labs for patient 12345 and identify any critical values or concerning trends
```

**Medication Safety:**
```
medster>> Review the medication list for patient 12345 and check for potential drug interactions
```

**Differential Diagnosis:**
```
medster>> Patient presents with fatigue, weight loss, and night sweats. Generate a prioritized differential and optimal workup sequence.
```

## Available Tools

### Medical Data (SYNTHEA/FHIR)
- `get_patient_labs` - Laboratory results with reference ranges
- `get_vital_signs` - Vital sign measurements and trends
- `get_demographics` - Patient demographic information
- `get_clinical_notes` - Progress notes, H&P, consultations
- `get_soap_notes` - SOAP-formatted progress notes
- `get_discharge_summary` - Hospital discharge summaries
- `get_medication_list` - Current and historical medications
- `check_drug_interactions` - Drug-drug interaction screening
- `get_radiology_reports` - Imaging studies and interpretations

### Clinical Scores
- `calculate_clinical_score` - Wells' Criteria, CHA2DS2-VASc, CURB-65, MELD, etc.

### Complex Analysis
- `analyze_complex_note` - Multi-step clinical reasoning via MCP server (Claude/Anthropic)

## Data Sources

### Coherent Data Set
Medster uses the **Coherent Data Set** - a comprehensive synthetic dataset that includes:
- FHIR resources (patient records, labs, vitals, medications, notes)
- DICOM images (X-rays, CT scans)
- Genomic data
- Physiological data (ECGs)
- Clinical notes

All data types are linked together via FHIR references.

**Download**: https://synthea.mitre.org/downloads (9 GB)

**Citation**:
> Walonoski J, et al. The "Coherent Data Set": Combining Patient Data and Imaging in a Comprehensive, Synthetic Health Record. Electronics. 2022; 11(8):1199. https://doi.org/10.3390/electronics11081199

### MCP Medical Analysis Server
For complex note analysis, Medster integrates with your FastMCP medical analysis server that uses Claude/Anthropic for sophisticated clinical reasoning.

## Configuration

1. Download and extract the Coherent Data Set
2. Set these environment variables in your `.env` file:

```bash
ANTHROPIC_API_KEY=your_key           # Required for Claude Sonnet 4.5
COHERENT_DATA_PATH=./coherent_data/fhir  # Path to extracted FHIR data
MCP_SERVER_URL=http://localhost:8000  # Your MCP server
```

## Safety & Disclaimer

**IMPORTANT**: Medster is for research and educational purposes only.

- Not intended for clinical decision-making without physician review
- Always verify findings with appropriate clinical resources
- Critical values and drug interactions are simplified checks
- Use clinical judgment for all patient care decisions

## Architecture Details

Medster preserves Dexter's proven multi-agent architecture:

1. **Planning Module** - Decomposes clinical queries into tasks
2. **Action Module** - Selects appropriate tools for data retrieval
3. **Validation Module** - Verifies task completion
4. **Synthesis Module** - Generates comprehensive clinical analysis

Safety mechanisms include:
- Global step limits (default: 20)
- Per-task step limits (default: 5)
- Loop detection (prevents repetitive actions)
- Critical value flagging

## License

MIT License

## Acknowledgments

- [Dexter](https://github.com/virattt/dexter) by @virattt - The original financial research agent that inspired this architecture. Medster adapts Dexter's proven multi-agent loop (planning → action → validation → synthesis) for the medical domain. A local reference copy of the Dexter codebase is maintained in `dexter-reference/` for architectural consultation during development.
- [SYNTHEA](https://synthetichealth.github.io/synthea/) - Synthetic patient data generator
- [HAPI FHIR](https://hapifhir.io/) - FHIR server implementation
- [Coherent Data Set](https://synthea.mitre.org/downloads) - 9GB synthetic dataset integrating FHIR, DICOM, genomics, and ECG data for comprehensive multimodal medical AI research
