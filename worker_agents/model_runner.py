import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
ENV_FILE_PATH = UTILS_DIR / "var.env"

load_dotenv(ENV_FILE_PATH)

@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION"),
    )

""" helper function to be used by the agents to run the LLM """
def run_model(
        *,
        instructions: str,
        contents: List[types.Content],
        tools: Optional[List[types.Tool]] = None,
        model: str = None,
        thinking_budget: Optional[int] = None,
) -> types.GenerateContentResponse:
    client = get_genai_client()
    config_kwargs: dict[str, Any] = {"system_instruction": instructions}
    if tools:
        config_kwargs["tools"] = tools
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    return client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )
