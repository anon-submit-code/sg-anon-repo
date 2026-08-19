from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class SubPrompt(BaseModel):
    id: int
    content: str
    opaque_values: dict[str, str] = Field(default_factory=dict)
    suggested_tools: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    original_prompt: str
    sub_prompts: list[SubPrompt]


@dataclass
class OpaqueValues:
    redacted: Any
    secrets: dict[str, str]
