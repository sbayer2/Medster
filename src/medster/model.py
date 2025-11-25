import os
import time
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Type, List, Optional, Union, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage

from medster.prompts import DEFAULT_SYSTEM_PROMPT


def call_llm(
    prompt: str,
    model: str = "claude-sonnet-4.5",
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[BaseTool]] = None,
    images: Optional[List[str]] = None,
) -> AIMessage:
    """
    Call Claude LLM with the given prompt and configuration.

    Args:
        prompt: The user prompt to send
        model: The Claude model to use (default: claude-sonnet-4.5)
               Supports: claude-sonnet-4.5, claude-opus-4.5, claude-haiku-4
               Sonnet 4.5: Balanced performance for clinical analysis
        system_prompt: Optional system prompt override
        output_schema: Optional Pydantic schema for structured output
        tools: Optional list of tools to bind
        images: Optional list of base64-encoded PNG images for vision analysis

    Returns:
        AIMessage or structured output based on schema
    """
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    # Map model names to Anthropic model IDs
    model_mapping = {
        "claude-sonnet-4.5": "claude-sonnet-4-20250514",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "claude-opus-4.5": "claude-opus-4-5-20251101",
        "claude-haiku-4": "claude-haiku-4-20250615",
    }

    anthropic_model = model_mapping.get(model, "claude-sonnet-4-20250514")

    # Initialize Claude LLM
    llm = ChatAnthropic(
        model=anthropic_model,
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Add structured output or tools to the LLM
    runnable = llm
    if output_schema:
        runnable = llm.with_structured_output(output_schema, method="function_calling")
    elif tools:
        runnable = llm.bind_tools(tools)

    # Build messages based on whether images are included
    if images:
        # Multimodal message with images
        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        # Add each image to content
        for img_base64 in images:
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_base64
                }
            })

        # Create multimodal message
        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": content_parts}
        ]

        # Retry logic for transient connection errors
        for attempt in range(3):
            try:
                return runnable.invoke(messages)
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s backoff

    else:
        # Text-only message (original behavior)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", final_system_prompt),
            ("user", "{prompt}")
        ])

        chain = prompt_template | runnable

        # Retry logic for transient connection errors
        for attempt in range(3):
            try:
                return chain.invoke({"prompt": prompt})
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s backoff
