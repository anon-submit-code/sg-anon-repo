from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

from shardguard.core.prompts import TOOL_PROMPT

logger = logging.getLogger(__name__)

_LLM_CTX_LOG = Path("shardguard_llm_context.log")


def _write_context_log(entry: dict[str, Any]) -> None:
    line = json.dumps(entry, ensure_ascii=False, default=str)
    try:
        with _LLM_CTX_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("Failed to write LLM context log: %s", exc)


@runtime_checkable
class ToolArgsLLM(Protocol):
    async def generate_tool_args(
        self,
        *,
        tool_name: str,
        tool_schema: dict[str, Any],
        tool_description: str | None = None,
        step_text: str,
        visible_placeholders: set[str],
        previous_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def _build_executor_prompt(
    *,
    tool_name: str,
    step_text: str,
    visible_placeholders: set[str],
    previous_results: dict[str, Any] | None,
) -> tuple[str, str]:
    """Returns (system, user) prompts."""

    sys = TOOL_PROMPT

    ph_line = ""
    if visible_placeholders:
        ph_line = "Allowed placeholders (use exactly as written):\n" + "\n".join(
            f"$[{k}]" for k in sorted(visible_placeholders)
        )

    prev_block = ""
    if previous_results:
        prev_block = "Previous step results (redacted):\n" + json.dumps(
            previous_results, indent=2, ensure_ascii=False
        )

    user = "\n\n".join(
        x
        for x in [
            f"Step:\n{step_text}".strip(),
            ph_line.strip(),
            prev_block.strip(),
            f"Return ONLY the JSON arguments object for `{tool_name}`.",
        ]
        if x
    )

    return sys, user


_UNSUPPORTED_SCHEMA_KEYS = frozenset({"$schema", "$id", "$comment", "examples", "definitions", "$defs"})


def _schema_has_open_object(schema: dict[str, Any]) -> bool:
    # True if any object has no declared properties, incompatible with OpenAI strict mode.
    if not isinstance(schema, dict):
        return False
    if schema.get("type") == "object" and not schema.get("properties"):
        return True
    for v in (schema.get("properties") or {}).values():
        if _schema_has_open_object(v):
            return True
    return False


def _normalize_schema_for_strict(schema: dict[str, Any]) -> dict[str, Any]:
    # Without this, strict mode silently degrades.
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
    if out.get("type") != "object":
        return out
    props = out.get("properties")
    if not props:
        return out
    out["required"] = list(props.keys())
    out["additionalProperties"] = False
    # Recursively normalize nested object schemas
    out["properties"] = {
        k: (_normalize_schema_for_strict(v) if isinstance(v, dict) else v)
        for k, v in props.items()
    }
    return out


@dataclass
class OpenAIResponsesToolArgsLLM:
    openai_client: Any
    model: str

    async def generate_tool_args(
        self,
        *,
        tool_name: str,
        tool_schema: dict[str, Any],
        tool_description: str | None = None,
        step_text: str,
        visible_placeholders: set[str],
        previous_results: dict[str, Any] | None = None,
        usage_stats: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        sys, user = _build_executor_prompt(
            tool_name=tool_name,
            step_text=step_text,
            visible_placeholders=visible_placeholders,
            previous_results=previous_results,
        )

        # OpenAI requires ^[a-zA-Z0-9_-]+$ — strip server prefix if present
        api_name = tool_name.split(".")[-1] if "." in tool_name else tool_name
        use_strict = not _schema_has_open_object(tool_schema)
        tool_def = {
            "type": "function",
            "name": api_name,
            "description": tool_description or f"Tool `{tool_name}`",
            "parameters": _normalize_schema_for_strict(tool_schema) if use_strict else tool_schema,
            "strict": use_strict,
        }

        _write_context_log(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "llm_instance": "executor_openai",
                "model": self.model,
                "tool_name": tool_name,
                "context": {
                    "system": sys,
                    "user": user,
                    "tool_schema": tool_schema,
                },
                "estimated_chars": len(sys) + len(user),
            }
        )

        resp = await self.openai_client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            tools=[tool_def],
            tool_choice={"type": "function", "name": api_name},
            parallel_tool_calls=False,
            temperature=0,
        )

        if usage_stats is not None:
            u = getattr(resp, "usage", None)
            if u:
                usage_stats["prompt_tokens"] += getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0) or 0
                usage_stats["completion_tokens"] += getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", 0) or 0
                usage_stats["total_tokens"] += getattr(u, "total_tokens", 0) or 0

        fc = next(
            (
                it
                for it in (getattr(resp, "output", None) or [])
                if getattr(it, "type", None) == "function_call"
            ),
            None,
        )
        if not fc:
            raise RuntimeError(
                f"Executor did not return a function_call for {tool_name}"
            )

        logger.debug(
            "Executor(OpenAI) function_call arguments: %s",
            getattr(fc, "arguments", None),
        )
        return json.loads(getattr(fc, "arguments", "") or "{}")


@dataclass
class GeminiToolArgsLLM:
    # Uses Google GenAI SDK with JSON schema output.

    gemini_async_client: Any
    model: str

    async def generate_tool_args(
        self,
        *,
        tool_name: str,
        tool_schema: dict[str, Any],
        tool_description: str | None = None,
        step_text: str,
        visible_placeholders: set[str],
        previous_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sys, user = _build_executor_prompt(
            tool_name=tool_name,
            step_text=step_text,
            visible_placeholders=visible_placeholders,
            previous_results=previous_results,
        )
        prompt = sys + "\n\n" + user

        _write_context_log(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "llm_instance": "executor_gemini",
                "model": self.model,
                "tool_name": tool_name,
                "context": {
                    "prompt": prompt,
                    "tool_schema": tool_schema,
                },
                "estimated_chars": len(prompt),
            }
        )

        resp = await self.gemini_async_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": tool_schema,
                "temperature": 0,
            },
        )

        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, dict):
            return parsed

        txt = getattr(resp, "text", "") or ""
        return json.loads(txt)


@dataclass
class OllamaToolArgsLLM:

    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    timeout_s: float = 60.0

    async def generate_tool_args(
        self,
        *,
        tool_name: str,
        tool_schema: dict[str, Any],
        tool_description: str | None = None,
        step_text: str,
        visible_placeholders: set[str],
        previous_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sys, user = _build_executor_prompt(
            tool_name=tool_name,
            step_text=step_text,
            visible_placeholders=visible_placeholders,
            previous_results=previous_results,
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            "format": tool_schema,
            "options": {"temperature": 0},
        }

        _write_context_log(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "llm_instance": "executor_ollama",
                "model": self.model,
                "tool_name": tool_name,
                "context": {
                    "messages": payload["messages"],
                    "tool_schema": tool_schema,
                },
                "estimated_chars": len(sys) + len(user),
            }
        )

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()

        msg = (
            (data.get("message") or {}).get("content")
            if isinstance(data, dict)
            else None
        )
        if not isinstance(msg, str):
            raise RuntimeError(f"Ollama response missing message.content: {data}")

        return json.loads(msg)
