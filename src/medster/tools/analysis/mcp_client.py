"""
MCP Medical Analysis Client

Connects to FastMCP medical analysis server for specialist-level clinical document analysis.

Recursive AI Architecture:
- Local Agent: Claude Sonnet 4.5 (Medster) - Orchestration, tool selection, FHIR data extraction
- Remote Server: Claude Sonnet 4.5 (FastMCP) - Specialist medical document analysis

This creates a "medical specialist consultant" in Medster's backpack that can be delegated
complex clinical reasoning tasks requiring deep medical knowledge and multi-step analysis.
"""

from langchain.tools import tool
from typing import Literal, Optional
from pydantic import BaseModel, Field
import os
import requests
import json
import sys

####################################
# MCP Server Configuration
####################################

# Your FastMCP medical analysis server endpoint
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

# MCP API Key for authentication
MCP_API_KEY = os.getenv("MCP_API_KEY", "")

# Enable debug logging for MCP calls
MCP_DEBUG = os.getenv("MCP_DEBUG", "false").lower() == "true"

# Debug log file
MCP_LOG_FILE = "/Users/sbm4_mac/Desktop/Medster/mcp_debug.log"

# Synthetic data disclaimer for Claude safeguards
SYNTHETIC_DATA_DISCLAIMER = """
[DISCLAIMER: This is SYNTHETIC patient data from the Coherent Data Set (SYNTHEA).
This is NOT real patient data - no PHI or HIPAA concerns apply.
This data is generated for medical AI research and education purposes.
Source: https://synthea.mitre.org/downloads - Coherent Data Set]

"""

def mcp_log(message: str):
    """Write debug message to log file"""
    if MCP_DEBUG:
        with open(MCP_LOG_FILE, "a") as f:
            f.write(f"{message}\n")
        # Also try stderr which might not be captured
        print(f"{message}", file=sys.stderr)


####################################
# Input Schemas
####################################

class ComplexNoteAnalysisInput(BaseModel):
    note_text: str = Field(description="The clinical note text to analyze (SOAP note, discharge summary, consult note, etc.)")
    analysis_type: Literal["basic", "comprehensive", "complicated"] = Field(
        default="complicated",
        description="Level of analysis: 'basic' for simple extraction, 'comprehensive' for detailed analysis, 'complicated' for multi-step clinical reasoning with quality assurance"
    )
    # NOTE: context parameter removed - not supported by deployed FastMCP server
    # Context can be prepended to note_text if needed


####################################
# Tools
####################################

@tool(args_schema=ComplexNoteAnalysisInput)
def analyze_medical_document(
    note_text: str,
    analysis_type: Literal["basic", "comprehensive", "complicated"] = "complicated"
) -> dict:
    """
    Analyzes medical documents using the FastMCP server with Claude Sonnet 4.5.
    Delegates complex clinical analysis to the MCP medical server for AI-powered insights.

    Analysis types:
    - basic: Quick extraction of key clinical data
    - comprehensive: Detailed analysis with clinical context (multi-step reasoning)
    - complicated: Alias for 'comprehensive' (automatically mapped on client side)

    Note: 'complicated' is automatically mapped to 'comprehensive' when calling the server,
    as the deployed FastMCP server uses 'comprehensive' for advanced analysis with Claude Sonnet 4.5.

    Useful for: SOAP notes, discharge summaries, lab interpretations,
    clinical pattern recognition, and decision support.

    Architecture: This tool creates a recursive AI system where:
    - Local: Claude Sonnet 4.5 (Medster) handles orchestration and tool selection
    - Remote: Claude Sonnet 4.5 (MCP Server) provides specialist medical analysis
    """
    try:
        mcp_log(f"[MCP] Calling server at {MCP_SERVER_URL}")
        mcp_log(f"[MCP] Analysis type: {analysis_type}")
        mcp_log(f"[MCP] Note text length: {len(note_text)} chars")

        # Map Medster analysis types to server analysis types
        # Server uses "comprehensive" for advanced analysis (not "complicated")
        server_analysis_type = analysis_type
        if analysis_type == "complicated":
            server_analysis_type = "comprehensive"
            mcp_log(f"[MCP] Mapping 'complicated' -> 'comprehensive' for server")

        # Prepend synthetic data disclaimer to avoid Claude safeguard issues
        # The Coherent Data Set is synthetic - no PHI concerns
        note_with_disclaimer = SYNTHETIC_DATA_DISCLAIMER + note_text
        mcp_log(f"[MCP] Added synthetic data disclaimer ({len(SYNTHETIC_DATA_DISCLAIMER)} chars)")

        # Build MCP JSON-RPC request for tool call
        # FastMCP servers use JSON-RPC 2.0 protocol
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "analyze_medical_document",
                "arguments": {
                    "document_content": note_with_disclaimer,
                    "analysis_type": server_analysis_type,
                }
            }
        }

        # NOTE: The deployed FastMCP server does not support the 'context' parameter
        # Context can be prepended to the document_content if needed
        # if context:
        #     mcp_request["params"]["arguments"]["context"] = context

        # Try MCP JSON-RPC endpoint - use URL directly as provided
        mcp_endpoints = [
            MCP_SERVER_URL,  # Use the complete endpoint URL from config
        ]

        last_error = None
        for endpoint in mcp_endpoints:
            try:
                mcp_log(f"[MCP] Trying endpoint: {endpoint}")
                if endpoint.endswith("/mcp") or endpoint.endswith("/rpc"):
                    mcp_log(f"[MCP] Request params: {mcp_request['params']['arguments']}")

                # Build headers with optional auth
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
                if MCP_API_KEY:
                    headers["Authorization"] = f"Bearer {MCP_API_KEY}"

                # Determine if this is MCP or REST endpoint
                if endpoint.endswith("/analyze_medical_document"):
                    # REST endpoint - use simple payload with disclaimer
                    payload = {
                        "document": note_with_disclaimer,
                        "analysis_type": server_analysis_type,  # Use mapped type
                    }
                    # NOTE: context parameter removed - not supported by FastMCP server
                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=120
                    )
                else:
                    # MCP JSON-RPC endpoint
                    mcp_log(f"[MCP] Sending JSON-RPC request to {endpoint}")
                    response = requests.post(
                        endpoint,
                        json=mcp_request,
                        headers=headers,
                        timeout=120
                    )
                    mcp_log(f"[MCP] Response status: {response.status_code}")
                    mcp_log(f"[MCP] Response headers: {dict(response.headers)}")
                    mcp_log(f"[MCP] Response body (first 500 chars): {response.text[:500]}")

                if response.status_code == 200:
                    # Handle SSE (Server-Sent Events) format from FastMCP
                    response_text = response.text
                    mcp_log(f"[MCP] Parsing response format")

                    # Parse SSE format: SSE can have ping comments (: ping), event lines, and data lines
                    # Look for "data:" line which contains the JSON-RPC response
                    if "event:" in response_text or response_text.startswith(":"):
                        # Extract JSON from SSE data line
                        lines = response_text.split("\n")
                        mcp_log(f"[MCP] Found {len(lines)} lines in SSE response")
                        for line in lines:
                            if line.startswith("data:"):
                                json_str = line[5:].strip()  # Remove "data:" prefix
                                mcp_log(f"[MCP] Found data line, length: {len(json_str)} chars")
                                result = json.loads(json_str)
                                mcp_log(f"[MCP] Successfully parsed JSON-RPC response")
                                break
                        else:
                            result = {"error": "No data in SSE response"}
                    else:
                        # Regular JSON response
                        mcp_log(f"[MCP] Parsing as regular JSON")
                        result = response.json()

                    mcp_log(f"[MCP] Success from {endpoint}")
                    mcp_log(f"[MCP] Response keys: {result.keys() if isinstance(result, dict) else 'not dict'}")

                    if isinstance(result, dict) and "result" in result:
                        mcp_log(f"[MCP] Result keys: {result['result'].keys() if isinstance(result['result'], dict) else type(result['result'])}")

                    # Handle MCP JSON-RPC response format
                    if "result" in result:
                        # MCP response - extract content
                        mcp_result = result["result"]
                        if isinstance(mcp_result, dict) and "content" in mcp_result:
                            # Extract text content from MCP response
                            content = mcp_result["content"]
                            mcp_log(f"[MCP] Content type: {type(content)}, length: {len(content) if isinstance(content, (list, str)) else 'N/A'}")

                            if isinstance(content, list) and len(content) > 0:
                                analysis_text = content[0].get("text", str(content))
                            else:
                                analysis_text = str(content)

                            mcp_log(f"[MCP] Analysis text length: {len(analysis_text)} chars")

                            # Check if this is an error response
                            if mcp_result.get("isError"):
                                mcp_log(f"[MCP] ERROR RESPONSE: {analysis_text}")
                                return {
                                    "analysis_type": analysis_type,
                                    "server_analysis_type": server_analysis_type,
                                    "status": "error",
                                    "error": f"MCP Server Error: {analysis_text}",
                                    "source": f"MCP Medical Analysis Server ({endpoint})"
                                }

                            return {
                                "analysis_type": analysis_type,
                                "server_analysis_type": server_analysis_type,
                                "status": "success",
                                "analysis": analysis_text,
                                "tokens_used": mcp_result.get("tokens_used", 0),
                                "source": f"MCP Medical Analysis Server ({endpoint})"
                            }
                        else:
                            # No content field - return raw result
                            mcp_log(f"[MCP] No 'content' field in result, returning raw: {str(mcp_result)[:200]}")
                            return {
                                "analysis_type": analysis_type,
                                "server_analysis_type": server_analysis_type,
                                "status": "success",
                                "analysis": mcp_result,
                                "source": f"MCP Medical Analysis Server ({endpoint})"
                            }
                    elif "error" in result:
                        # MCP error response
                        last_error = result["error"].get("message", str(result["error"]))
                        continue
                    else:
                        # REST response format
                        return {
                            "analysis_type": analysis_type,
                            "server_analysis_type": server_analysis_type,
                            "status": "success",
                            "analysis": result.get("analysis", result),
                            "tokens_used": result.get("tokens_used", 0),
                            "source": f"MCP Medical Analysis Server ({endpoint})"
                        }
                elif response.status_code == 404:
                    last_error = f"Endpoint not found: {endpoint}"
                    mcp_log(f"[MCP] 404 response body: {response.text[:500]}")
                    continue
                else:
                    last_error = f"Status {response.status_code}: {response.text[:200]}"
                    mcp_log(f"[MCP] Error response: Status {response.status_code}")
                    mcp_log(f"[MCP] Response body: {response.text[:500]}")
                    continue

            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection failed to {endpoint}"
                continue
            except Exception as e:
                last_error = str(e)
                continue

        # All endpoints failed
        mcp_log(f"[MCP] All endpoints failed. Last error: {last_error}")
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error": f"All MCP endpoints failed. Last error: {last_error}",
            "server_url": MCP_SERVER_URL,
            "recommendation": "Check MCP server status and endpoint configuration"
        }

    except requests.exceptions.Timeout:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error": "MCP server request timed out",
            "recommendation": "Try 'comprehensive' or 'basic' analysis type for faster results"
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error": str(e)
        }
