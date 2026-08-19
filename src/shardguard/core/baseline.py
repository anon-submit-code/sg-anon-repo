"""
baseline.py — for running without shardguard.

Single-LLM agent: one OpenAI chat-completions call with all registered MCP tools
visible. No redaction, no planner, no per-step secret scoping. Used to compare
against the ShardGuard.

Writes three separate log files (parallel to ShardGuard's logs) so runs can be
analyzed and compared side-by-side:
  baseline_debug.log        ↔  shardguard_debug.log
  baseline_audit.log        ↔  shardguard_privacy_audit.log
  baseline_llm_context.log  ↔  shardguard_llm_context.log
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from shardguard.core.coordination import McpTool, McpToolCatalog
from shardguard.core.execution import McpToolExecutor

_BASELINE_AUDIT_LOG = Path("baseline_audit.log")
_BASELINE_LLM_CTX_LOG = Path("baseline_llm_context.log")

# baseline_debug.log via standard Python logging (separate handler added below)
_baseline_file_handler = logging.FileHandler("baseline_debug.log", mode="w")
_baseline_file_handler.setLevel(logging.DEBUG)
_baseline_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

logger = logging.getLogger(__name__)
logger.addHandler(_baseline_file_handler)
logger.setLevel(logging.DEBUG)

_MAX_ITERATIONS = 20


def _write_audit(event: dict[str, Any]) -> None:
    line = json.dumps(event, ensure_ascii=False)
    try:
        with _BASELINE_AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.error("Failed to write baseline audit log: %s", e)


def _write_context_log(entry: dict[str, Any]) -> None:
    line = json.dumps(entry, ensure_ascii=False, default=str)
    try:
        with _BASELINE_LLM_CTX_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.warning("Failed to write baseline LLM context log: %s", e)


def _safe_fn_name(name: str) -> str:
    return name.replace(".", "__")


def _simplify_schema(schema: Any) -> Any:
    # Strip null from anyOf only for numeric types — MCP servers reject null numbers even when schema allows them.
    if not isinstance(schema, dict):
        return schema
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s != {"type": "null"} and s.get("type") != "null"]
        if len(non_null) == 1 and non_null[0].get("type") in ("number", "integer"):
            merged = {**schema, **non_null[0]}
            merged.pop("anyOf")
            return _simplify_schema(merged)
    out = {}
    for k, v in schema.items():
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _simplify_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _simplify_schema(v)
        else:
            out[k] = v
    return out


def _to_openai_tool(tool: McpTool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _safe_fn_name(tool.name),
            "description": tool.description,
            "parameters": _simplify_schema(tool.parameters),
        },
    }


class BaselineAgent:
    def __init__(
        self,
        registry_path: str,
        openai_model: str = "gpt-4o-mini",
        registered_mcps: list[str] | None = None,
        base_url: str | None = None,
    ) -> None:
        self.registry_path = registry_path
        self.openai_model = openai_model
        self.registered_mcps = registered_mcps
        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url if base_url.endswith("/v1") else base_url + "/v1"
            kwargs["api_key"] = "ollama"
        self._openai = AsyncOpenAI(**kwargs)
        self._executor = McpToolExecutor(registry_path)

    def _audit(
        self,
        *,
        phase: str,
        run_id: str,
        iteration: int | None,
        tool: str | None,
        event: dict[str, Any],
    ) -> None:
        record = {
            "event_id": str(uuid.uuid4()),
            "run_id": run_id,
            "phase": phase,
            "iteration": iteration,
            "tool": tool,
            **event,
        }
        _write_audit(record)

    async def run(self, prompt: str) -> dict[str, Any]:
        run_id = str(uuid.uuid4())

        catalog = McpToolCatalog(self.registry_path, registered_mcps=self.registered_mcps)
        catalog.refresh()
        if self.registered_mcps:
            all_tools = list(catalog.all_tools())
        else:
            _STDIO_SKIP = {"local-file-mcp", "github-reader", "stripe-sandbox"}
            all_tools = [t for t in catalog.all_tools() if t.server not in _STDIO_SKIP]


        openai_tools = [_to_openai_tool(t) for t in all_tools]
        tool_map: dict[str, McpTool] = {_safe_fn_name(t.name): t for t in all_tools}

        system_msg = (
            "You are a helpful assistant with access to tools. Use them to complete the user's request. "
            "When reading local files, the allowed directory is /Users/xx/Documents/local-folder — "
            "always use full absolute paths (e.g. /Users/xx/Documents/local-folder/text.txt)."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        trace: list[dict[str, Any]] = []
        usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        final_text = ""

        self._audit(
            phase="job_start",
            run_id=run_id,
            iteration=None,
            tool=None,
            event={
                "message": "Baseline loop started",
                "model": self.openai_model,
                "tools_available": [t.name for t in all_tools],
                "input_length": len(prompt),
                "rules_applied": [],
                "privacy_properties": [],
            },
        )

        for iteration in range(1, _MAX_ITERATIONS + 1):
            kwargs: dict[str, Any] = {"model": self.openai_model, "messages": messages}
            if openai_tools:
                kwargs["tools"] = openai_tools
                kwargs["tool_choice"] = "auto"

            _write_context_log(
                {
                    "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
                    "llm_instance": "baseline",
                    "model": self.openai_model,
                    "run_id": run_id,
                    "iteration": iteration,
                    "context": {
                        "messages": messages,
                        "tools_count": len(openai_tools),
                    },
                    "estimated_chars": sum(
                        len(json.dumps(m, ensure_ascii=False)) for m in messages
                    ),
                }
            )

            logger.debug("LLM Call #%d: iteration %d", iteration, iteration)

            response = await self._openai.chat.completions.create(**kwargs)

            u = response.usage
            if u:
                usage["prompt_tokens"] += u.prompt_tokens or 0
                usage["completion_tokens"] += u.completion_tokens or 0
                usage["total_tokens"] += u.total_tokens or 0

            choice = response.choices[0]
            msg = choice.message
            messages.append(msg.model_dump(exclude_unset=False))

            if not msg.tool_calls:
                final_text = msg.content or ""
                self._audit(
                    phase="job_complete",
                    run_id=run_id,
                    iteration=iteration,
                    tool=None,
                    event={
                        "message": "Baseline loop finished",
                        "total_tool_calls": len(trace),
                        "usage": usage,
                        "rules_applied": [],
                        "privacy_properties": [],
                    },
                )
                break

            for tc in msg.tool_calls:
                fn = tc.function
                try:
                    args = json.loads(fn.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                tool_info = tool_map.get(fn.name)
                server = tool_info.server if tool_info else fn.name
                qualified = tool_info.name if tool_info else fn.name
                original_tool_name = qualified.split(".")[-1] if "." in qualified else qualified

                logger.debug(
                    "LLM Call #%d: Execute iteration=%d via %s\n  args=%s",
                    iteration,
                    iteration,
                    original_tool_name,
                    json.dumps(args, ensure_ascii=False),
                )

                try:
                    result = self._executor.call(server, original_tool_name, args)
                except Exception as e:
                    result = {"error": str(e)}

                logger.debug(
                    "Tool output (%r):\n  %s",
                    fn.name,
                    json.dumps(result, ensure_ascii=False),
                )

                self._audit(
                    phase="tool_call",
                    run_id=run_id,
                    iteration=iteration,
                    tool=fn.name,
                    event={
                        "server": server,
                        "arguments": args,
                        "result_length": len(json.dumps(result, ensure_ascii=False)),
                        "rules_applied": [],
                        "privacy_properties": [],
                    },
                )

                trace.append(
                    {
                        "tool": fn.name,
                        "server": server,
                        "arguments": args,
                        "result": result,
                        "llm_context": json.dumps(messages, ensure_ascii=False),
                        "secrets_for_step": list(args.keys()),
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        else:
        """ In experiment runs, we never reach _MAX_ITERATIONS """
            final_text = f"Reached maximum iterations ({_MAX_ITERATIONS}) without a final answer."
            self._audit(
                phase="job_max_iterations",
                run_id=run_id,
                iteration=_MAX_ITERATIONS,
                tool=None,
                event={
                    "message": final_text,
                    "rules_applied": [],
                    "privacy_properties": [],
                },
            )

        return {
            "run_id": run_id,
            "final_text": final_text,
            "trace": trace,
            "usage": usage,
        }
