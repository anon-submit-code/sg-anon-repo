"""
coordination.py

Zero-trust coordinator that:

1. Delegates sensitive data detection and opaque placeholder substitution ($[KEY]) to a Redaction LLM
2. Plans tool execution steps using live MCP tool schemas
3. Runs each step with a per-step scoped vault, only placeholders authorized for that step are visible
4. Resolves placeholders to real values only at tool execution time
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI  # type: ignore

from shardguard.core.execution import McpToolExecutor
from shardguard.core.planning import PlanningLLM
from shardguard.core.prompts import FINAL_SUMMARY_PROMPT, REDACTION_PROMPT
from shardguard.core.redactor import LlmOpaqueRedactor
from shardguard.core.sanitization import (
    _DOLLAR_KEY_RE,
    _PLACEHOLDER_RE,
    _PLACEHOLDER_TOKEN_RE,
    _as_steps_list,
    _normalize_compound_tokens,
    _as_str_list,
    _build_tool_def,
    _canonicalize_scalar_placeholders,
    _coerce_types,
    _extract_placeholder_keys,
    _validate_formats,
)
from shardguard.core.tool_args_llm import (
    GeminiToolArgsLLM,
    OllamaToolArgsLLM,
    OpenAIResponsesToolArgsLLM,
    ToolArgsLLM,
    _build_executor_prompt,
    _write_context_log,
)
from shardguard.mcp_servers import registry

_AUDIT_LOG_PATH = Path("shardguard_privacy_audit.log")

logging.basicConfig(
    filename="shardguard_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",  # overwrite log file on each run
    force=True,
)

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _json_eq(a: Any, b: Any) -> bool:
    """Compare two JSON-compatible objects for equality."""
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(
        b, sort_keys=True, ensure_ascii=False
    )


# Oracle plan augmentation helpers, used for eval debugging

_ORACLE_ID_PARAMS: frozenset[str] = frozenset({
    "employee_id", "patient_id", "account_id", "contact_id", "customer_id",
    "policy_id", "vehicle_id", "booking_id", "record_id", "case_id",
    "claim_id", "order_id", "interaction_id", "agreement_id", "salary_id",
    "legal_record_id", "insurance_id", "traveler_id", "credit_score_id",
    "appointment_id",
})


def _oracle_is_id_param(name: str) -> bool:
    if name in _ORACLE_ID_PARAMS:
        return True
    lower = name.lower()
    return lower.endswith("_id") or lower.endswith("_ids")


def _oracle_is_email_param(name: str) -> bool:
    lower = name.lower()
    return lower in {"email", "email_address", "to", "to_email", "from_email", "recipient_email"} or "email" in lower


def _oracle_is_phone_param(name: str) -> bool:
    lower = name.lower()
    return lower in {"phone", "phone_number", "mobile", "cell"} or lower.startswith("phone_")


def _oracle_is_password_param(name: str) -> bool:
    lower = name.lower()
    return "password" in lower or lower in {"credential", "secret", "portal_password"}


def _append_privacy_audit(event: dict[str, Any]) -> None:
    """Writes a raw audit event to the log file."""
    line = json.dumps(event, ensure_ascii=False)
    try:
        _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.error(f"Failed to write to privacy audit log: {e}")


def _compute_rules_applied(
    *,
    tool_def: dict[str, Any] | None = None,
    secrets_for_step: dict[str, str] | None = None,
    prev_for_step: dict[str, Any] | None = None,
    previous_results: dict[str, Any] | None = None,
    parsed_args_before: dict[str, Any] | None = None,
    parsed_args_after: dict[str, Any] | None = None,
    resolved_args_before: dict[str, Any] | None = None,
    resolved_args_after: dict[str, Any] | None = None,
    format_errs: list[dict[str, Any]] | None = None,
    unresolved: list[str] | None = None,
    concat_placeholders: bool | None = None,
) -> list[str]:
    """Dynamically determine which privacy rules were enforced during the step."""
    rules: list[str] = []

    if tool_def and tool_def.get("strict") is True:
        rules.append("strict_schema")

    if secrets_for_step is not None:
        rules.append("per_step_secret_scope")

    if (
        prev_for_step is not None
        and previous_results is not None
        and set(prev_for_step.keys()) != set(previous_results.keys())
    ):
        rules.append("per_step_prev_results_scope")

    if (
        parsed_args_before is not None
        and parsed_args_after is not None
        and not _json_eq(parsed_args_before, parsed_args_after)
    ):
        rules.append("placeholder_canonicalization")

    if unresolved is not None:
        rules.append("unresolved_placeholder_check")
        if unresolved:
            rules.append("unresolved_placeholders_found")

    if concat_placeholders is not None:
        rules.append("concat_placeholder_check")
        if concat_placeholders:
            rules.append("concat_placeholders_found")

    if (
        resolved_args_before is not None
        and resolved_args_after is not None
        and not _json_eq(resolved_args_before, resolved_args_after)
    ):
        rules.append("type_coercion")

    if format_errs is not None:
        rules.append("format_validation")
        if format_errs:
            rules.append("format_validation_failed")

    return rules


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    parameters: dict[str, Any]
    server: str


def _augment_schema_for_placeholders(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return schema

    def wrap_scalar(prop: dict[str, Any]) -> dict[str, Any]:
        if isinstance(prop.get("anyOf"), list):
            for opt in prop["anyOf"]:
                if (
                    isinstance(opt, dict)
                    and opt.get("type") == "string"
                    and opt.get("pattern") == _PLACEHOLDER_TOKEN_RE.pattern
                ):
                    return prop

            prop = dict(prop)
            prop["anyOf"] = [
                augment(o) for o in prop["anyOf"] if isinstance(o, dict)
            ] + [o for o in prop["anyOf"] if not isinstance(o, dict)]
            return prop

        t = prop.get("type")
        fmt = prop.get("format")
        is_scalar = (t in ("string", "integer", "number", "boolean")) or (
            fmt in ("email", "date-time")
        )
        if not is_scalar:
            return augment(prop)

        original = dict(prop)
        placeholder_opt = {"type": "string", "pattern": _PLACEHOLDER_TOKEN_RE.pattern}
        return {"anyOf": [original, placeholder_opt]}

    def augment(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        out = dict(node)
        if out.get("type") == "object" and isinstance(out.get("properties"), dict):
            out["properties"] = {
                k: wrap_scalar(v) if isinstance(v, dict) else v
                for k, v in out["properties"].items()
            }
            return out
        if out.get("type") == "array" and isinstance(out.get("items"), dict):
            out["items"] = augment(out["items"])
            return out
        for key in ("anyOf", "oneOf", "allOf"):
            if isinstance(out.get(key), list):
                out[key] = [augment(x) for x in out[key]]
        return out

    return augment(schema)


def _normalize_json_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    out = dict(schema)
    if out.get("type") != "object":
        out = {
            "type": "object",
            "properties": {"input": out},
            "required": ["input"],
            "additionalProperties": False,
        }

    out.setdefault("properties", {})
    original_required: set[str] = set(out.get("required") or [])

    props = {}
    for key, prop in (out.get("properties") or {}).items():
        if key not in original_required and isinstance(prop, dict):
            existing_any = prop.get("anyOf")
            already_nullable = isinstance(existing_any, list) and any(
                isinstance(o, dict) and o.get("type") == "null" for o in existing_any
            )
            if not already_nullable:
                props[key] = {"anyOf": [prop, {"type": "null"}]}
            else:
                props[key] = prop
        else:
            props[key] = prop
    out["properties"] = props

    out["additionalProperties"] = False
    out = _augment_schema_for_placeholders(out)
    out["required"] = list(out.get("properties", {}).keys())

    return out


class McpToolCatalog:
    def __init__(
        self,
        registry_path: str,
        *,
        init: bool = True,
        timeout: float = 15.0,
        registered_mcps: list[str] | None = None,
    ) -> None:
        self.registry_path = registry_path
        self.init = init
        self.timeout = timeout
        self.registered_mcps = registered_mcps

        self._tools: dict[str, McpTool] = {}

    def refresh(self) -> None:
        reg_data = registry.load_registry(self.registry_path)
        mcps_config = reg_data.get(registry.REG_MCP_KEY, {})
        if self.registered_mcps is not None:
            mcps_config = {k: v for k, v in mcps_config.items() if k in self.registered_mcps}

        tools_by_server = registry.fetch_all_tools(
            self.registry_path, init=self.init, timeout=self.timeout
        )
        if self.registered_mcps is not None:
            tools_by_server = {k: v for k, v in tools_by_server.items() if k in self.registered_mcps}

        flat: dict[str, McpTool] = {}

        for server_name, tools in tools_by_server.items():
            cfg = mcps_config.get(server_name, {})
            transport = cfg.get("transport", "")

            if transport == "stdio":
                stdio_args = cfg.get("stdio", {}).get("args", [])
                file_roots = [
                    a for a in stdio_args if isinstance(a, str) and a.startswith("/")
                ]
            else:
                client = registry.get_or_create_client(self.registry_path, server_name)
                file_roots = [
                    r["uri"].replace("file://", "")
                    for r in (client.resources_list(timeout=self.timeout) or [])
                    if isinstance(r.get("uri"), str) and r["uri"].startswith("file://")
                ]

            logger.debug(
                f"[catalog] server={server_name!r} transport={transport!r} "
                f"file_roots={file_roots}"
            )

            for t in tools or []:
                name = t.get("name") or t.get("title")
                if not name or not isinstance(name, str):
                    continue
                desc = t.get("description") or t.get("title") or ""
                schema = t.get("inputSchema") or t.get("parameters") or {}
                # Inject server root paths into description for tools with a path parameter
                if file_roots and "path" in (schema.get("properties") or {}):
                    desc = f"{desc}\nAllowed root paths: {', '.join(file_roots)}"
                params = _normalize_json_schema(schema)

                logger.debug(
                    f"[catalog] registered tool={server_name!r}.{name!r}\n"
                    f"  description={desc!r}\n"
                    f"  schema_props={list((schema.get('properties') or {}).keys())}"
                )

                qualified = f"{server_name}.{name}"
                tool = McpTool(
                    name=qualified,
                    description=str(desc),
                    parameters=params,
                    server=server_name,
                )
                flat[qualified] = tool
                if name not in flat:
                    flat[name] = tool
                elif flat[name].server != server_name:
                    del flat[name]

        self._tools = flat

    def get(self, tool_name: str) -> McpTool | None:
        return self._tools.get(tool_name)

    def all_tools(self) -> list[McpTool]:
        return [t for key, t in self._tools.items() if "." in key]


def _resolve_placeholders(value: Any, secrets: dict[str, str]) -> Any:
    if isinstance(value, str):

        def repl_bracket(m: re.Match[str]) -> str:
            key = m.group(1)
            return secrets.get(key, m.group(0))

        s = _PLACEHOLDER_RE.sub(repl_bracket, value)

        def repl_dollar(m: re.Match[str]) -> str:
            key = m.group(1)
            if key in secrets:
                return secrets[key]
            return m.group(0)

        s = _DOLLAR_KEY_RE.sub(repl_dollar, s)
        return s

    if isinstance(value, list):
        return [_resolve_placeholders(v, secrets) for v in value]

    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, secrets) for k, v in value.items()}

    return value


def _collect_unresolved_placeholders(value: Any, secrets: dict[str, str]) -> list[str]:
    unresolved: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, str):
            for m in _PLACEHOLDER_RE.finditer(x):
                key = m.group(1)
                if key not in secrets:
                    unresolved.add(key)
            for m in _DOLLAR_KEY_RE.finditer(x):
                key = m.group(1)
                if key not in secrets:
                    unresolved.add(key)
            return
        if isinstance(x, list):
            for i in x:
                walk(i)
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return

    walk(value)
    return sorted(unresolved)


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _iter_strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)


def _extract_id_from_structure(obj: Any, field_hint: str) -> str | None:
    hint_snake = field_hint.lower().replace(" ", "_")

    def search(node: Any) -> str | None:
        if isinstance(node, dict):
            for key, val in node.items():
                key_lower = key.lower()
                if key_lower == hint_snake or key_lower in hint_snake or hint_snake in key_lower:
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            if "id" in node and isinstance(node["id"], str) and node["id"].strip():
                return node["id"].strip()
            for v in node.values():
                result = search(v)
                if result:
                    return result
        elif isinstance(node, list) and node:
            return search(node[0])
        elif isinstance(node, str):
            stripped = node.strip()
            if stripped.startswith(("{", "[")):
                try:
                    return search(json.loads(stripped))
                except (json.JSONDecodeError, ValueError):
                    pass
            m = re.search(r"\$\[([A-Z_]+_\d+)\]", node)
            if m:
                return f"$[{m.group(1)}]"
        return None

    return search(obj)


def _contains_concat_placeholders(s: str) -> bool:
    return bool(re.search(r"(\$\[[A-Za-z0-9_]+\]|\$[A-Za-z][A-Za-z0-9_]*\$?){2,}", s))


def normalize_internal_id(value: object) -> object:
    if not isinstance(value, str):
        return value
    v = value.strip()
    v = v.strip('"').strip("'")
    v = v.rstrip(",.;:")
    v = v.rstrip("]} )")
    v = v.lstrip("[{(")
    v = v.strip()
    return v


class CoordinationService:

    def __init__(
        self,
        *,
        registry_path: str,
        openai_model: str = "gpt-4o-mini",
        step_executor: str | None = None,
        planning_model: str | None = None,
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
        gemini_model: str = "gemini-2.0-flash-exp",
        gemini_api_key: str | None = None,
        system_prompt: str = FINAL_SUMMARY_PROMPT,
        registered_mcps: list[str] | None = None,
        ablations: set[str] | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError(
                "openai package not installed; cannot create OpenAI client."
            )

        self.registry_path = registry_path
        self.openai_model = openai_model
        self.planning_model = planning_model or openai_model
        self.system_prompt = system_prompt
        self.ablations: set[str] = ablations or set()
        self._openai = AsyncOpenAI()
        self._exec_llm_step: ToolArgsLLM | None = None

        self._run_id: str = str(uuid.uuid4())
        self._prompt_id: str = str(uuid.uuid4())

        self._exec_llm_default: ToolArgsLLM = OpenAIResponsesToolArgsLLM(
            openai_client=self._openai,
            model=self.openai_model,
        )

        if step_executor == "ollama":
            self._exec_llm_step = OllamaToolArgsLLM(
                base_url=ollama_url,
                model=ollama_model,
            )
        elif step_executor == "gemini":
            if not gemini_api_key:
                raise ValueError(
                    "Gemini API key required for first_step_executor='gemini'"
                )
            self._exec_llm_step = GeminiToolArgsLLM(
                api_key=gemini_api_key,
                model=gemini_model,
            )

        self._redactor = LlmOpaqueRedactor(
            self._openai,
            model=self.openai_model,
            prompt_template=REDACTION_PROMPT,
        )
        self._planner = PlanningLLM(self._openai, model=self.planning_model)
        self._catalog = McpToolCatalog(registry_path, registered_mcps=registered_mcps)
        self._executor = McpToolExecutor(registry_path)
        self._catalog_ready = False

    async def _ensure_catalog(self) -> None:
        if not self._catalog_ready:
            await asyncio.to_thread(self._catalog.refresh)
            self._catalog_ready = True

    def _audit(
        self,
        *,
        phase: str,
        step_id: str | None,
        tool: str | None,
        recipient: str,
        event: dict[str, Any],
    ) -> None:
        """Internal audit helper that includes run/prompt IDs."""
        record = {
            "event_id": str(uuid.uuid4()),
            "run_id": self._run_id,
            "prompt_id": self._prompt_id,
            "phase": phase,
            "step_id": step_id,
            "tool": tool,
            "recipient": recipient,
            **event,
        }
        _append_privacy_audit(record)

    def _tool_summaries(self) -> list[tuple[str, str, list[str], list[str]]]:
        seen: set[str] = set()
        out = []
        for t in sorted(self._catalog._tools.values(), key=lambda x: x.name):
            if t.name in seen:
                continue
            seen.add(t.name)
            props = list((t.parameters.get("properties") or {}).keys())
            required = set(t.parameters.get("required") or [])
            req_params = [p for p in props if p in required]
            opt_params = [p for p in props if p not in required]
            out.append((t.name, t.description, req_params, opt_params))
        return out

    def _resolve_chain_tool(self, chain_name: str) -> str:
        if self._catalog.get(chain_name):
            return chain_name
        if "." not in chain_name:
            return chain_name  
        expected_server, short_name = chain_name.split(".", 1)
        tool = self._catalog.get(short_name)
        if not tool:
            return chain_name  
        if tool.server != expected_server:
            raise ValueError(
                f"Server mismatch: {chain_name!r} resolved to {short_name!r}, "
                f"but catalog tool belongs to server {tool.server!r}"
            )
        return short_name

    def _log_step_context(
        self, secrets: dict[str, str], previous_results: Any, step_desc: str
    ) -> None:
        opaque_vars_log = json.dumps(sorted(secrets.keys()), indent=2)
        prev_results_log = json.dumps(previous_results or {}, indent=2)

        logger.debug(
            f"\n### Available Opaque Variables (Use these as $VAR_NAME) ###\n"
            f"{opaque_vars_log}\n"
            f"### Previous Step Results ###\n"
            f"{prev_results_log}\n\n"
            f"### Task ###\n"
            f"{step_desc}"
        )

    def _record_step_error(
        self,
        trace: list[dict[str, Any]],
        previous_results: dict[str, Any],
        *,
        step_id: str,
        tool_name: str,
        error: Any,
    ) -> None:
        trace.append({"step": step_id, "tool": tool_name, "error": error})
        previous_results[step_id] = error

    def _select_arg_llm(self, idx: int) -> Any:
        if idx == 0 and self._exec_llm_step is not None:
            return self._exec_llm_step
        return self._exec_llm_default

    def _prev_for_step(
        self, step: dict[str, Any], previous_results: dict[str, Any]
    ) -> dict[str, Any]:
        deps = _as_str_list(step.get("reads", []) or step.get("depends_on", []))
        derived_args = step.get("derived_args") or {}
        if derived_args:
            needed_steps = {
                spec.get("from_step")
                for spec in derived_args.values()
                if isinstance(spec, dict)
            }
            return {sid: previous_results[sid] for sid in deps if sid in previous_results and sid in needed_steps}
        return {sid: previous_results[sid] for sid in deps if sid in previous_results}

    def _visible_for_step(
        self,
        *,
        task: str,
        step: dict[str, Any],
        prev_for_step: dict[str, Any],
        secrets: dict[str, str],
    ) -> tuple[dict[str, str], set[str]]:
        allowed_ph = _as_str_list(step.get("allowed_placeholders", []))

        allowed_key_set = set(allowed_ph)
        for k in (step.get("placeholder_args") or {}).keys():
            allowed_key_set.add(k)
        for k in _extract_placeholder_keys(task):
            allowed_key_set.add(k)
        for s in _iter_strings(step.get("args") or {}):
            for k in _extract_placeholder_keys(s):
                allowed_key_set.add(k)
        for s in _iter_strings(prev_for_step):
            for k in _extract_placeholder_keys(s):
                allowed_key_set.add(k)

        secrets_for_step = {k: v for k, v in secrets.items() if k in allowed_key_set}
        return secrets_for_step, set(secrets_for_step.keys())

    @staticmethod
    def _prefilled_args_from_step(
        step: dict[str, Any], *, visible_for_step: set[str]
    ) -> dict[str, Any]:
        mapping: dict[str, str] = step.get("placeholder_args") or {}
        from_placeholder_args = {arg_name: f"$[{ph_key}]" for ph_key, arg_name in mapping.items()}

        planner_args: dict[str, Any] = {}
        for k, v in (step.get("args") or {}).items():
            if isinstance(v, dict):
                continue  
            if isinstance(v, str) and _PLACEHOLDER_RE.search(v):
                keys_used = _extract_placeholder_keys(v)
                if all(key in visible_for_step for key in keys_used):
                    planner_args[k] = v
            else:
                planner_args[k] = v

        merged = {**planner_args, **from_placeholder_args}
        for arg_name in (step.get("derived_args") or {}).keys():
            merged.pop(arg_name, None)
        return merged

    @staticmethod
    def _prefill_derived_args(
        step: dict[str, Any], prev_for_step: dict[str, Any]
    ) -> dict[str, Any]:
        derived: dict[str, Any] = step.get("derived_args") or {}
        out: dict[str, Any] = {}
        for arg, info in derived.items():
            if not isinstance(info, dict):
                continue
            from_step = info.get("from_step")
            field_hint = info.get("field_hint", "")
            if not from_step or from_step not in prev_for_step:
                continue
            val = _extract_id_from_structure(prev_for_step[from_step], field_hint)
            if val:
                out[arg] = val
        return out

    @staticmethod
    def _augment_task_with_derived_args(
        task: str, step: dict[str, Any], already_prefilled: set[str] | None = None
    ) -> str:
        derived: dict[str, Any] = step.get("derived_args") or {}
        deps = _as_str_list(step.get("depends_on", []))

        if derived:
            hint_parts: list[str] = []
            for arg, info in derived.items():
                if not isinstance(info, dict):
                    continue
                if already_prefilled and arg in already_prefilled:
                    continue
                from_step = info.get("from_step", "prior step")
                field_hint = info.get("field_hint", "value")
                hint_parts.append(f'"{arg}" = {field_hint} from {from_step} result')

            if hint_parts:
                return f"{task}\nExtract these args from prior step results: {'; '.join(hint_parts)}"
            return task

        if deps:
"""
    Returning fallback hint to the executor LLM when the planner generated depends_on but left derived_args empty. 
    Without it the executor has no instruction to look at prior step results for IDs, so it might hallucinate argument values.
    The message tells it to look at what's already in the redacted results and use those placeholder keys rather than inventin new ones.
    It was called 422 times out of 9,898 executor calls during debugging/testing/development.
"""
            return (
                f"{task}\n"
                "Use only placeholder keys that appear in the prior step results (e.g. $[SECRET_1]). "
                "Do not invent placeholder values."
            )

        return task

    async def _get_tool_args_from_llm(
        self,
        *,
        idx: int,
        task: str,
        tool_def: dict[str, Any],
        tool_name: str,
        visible_for_step: set[str],
        prev_for_step: dict[str, Any],
        usage_stats: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        arg_llm = self._select_arg_llm(idx)
        return await McpToolExecutor._tool_executor(
            llm=arg_llm,
            tool_def=tool_def,
            tool_name=tool_name,
            step_text=task,
            visible_placeholders=visible_for_step,
            previous_results=prev_for_step,
            usage_stats=usage_stats,
        )

    async def _validate_and_resolve_args(
        self,
        *,
        parsed_args: dict[str, Any],
        tool_parameters: dict[str, Any],
        secrets_for_step: dict[str, str],
        visible_for_step: set[str],
        secrets_all: dict[str, str],
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        canon = _canonicalize_scalar_placeholders(
            parsed_args, tool_parameters, visible_for_step
        )

        unresolved = _collect_unresolved_placeholders(canon, secrets_for_step)
        concat = any(_contains_concat_placeholders(s) for s in _iter_strings(canon))
        if unresolved or concat:
            received = (
                await self._redactor.redact_obj(canon, prior_secrets=secrets_all)
            ).redacted
            err = {
                "error": "Unresolved placeholders or invalid placeholder formatting in tool arguments.",
                "unresolved": unresolved,
                "concat_placeholders": concat,
                "received": received,
            }
            return None, None, None, err

        resolved = _resolve_placeholders(canon, secrets_for_step)
        resolved = {k: v for k, v in resolved.items() if v is not None}
        resolved_pre_coerce = dict(resolved)
        resolved = _coerce_types(resolved, tool_parameters)

        format_errs = _validate_formats(resolved, tool_parameters)
        if format_errs:
            err = {
                "error": "Tool argument format validation failed.",
                "details": format_errs,
                "received": parsed_args,
            }
            return None, None, None, err

        return resolved, canon, resolved_pre_coerce, None

    async def _call_tool(
        self,
        *,
        server: str,
        tool_name: str,
        resolved_args: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                self._executor.call, server, tool_name, resolved_args
            )
        except Exception as e:
            return {"error": f"Tool execution failed: {e}"}

    async def _finalize_answer(
        self,
        *,
        redacted_prompt: str,
        final_payload: dict[str, Any],
        secrets: dict[str, str],
        usage_stats: dict[str, int],
    ) -> str:
        payload_text = (
            final_payload.get("text")
            if isinstance(final_payload, dict) and isinstance(final_payload.get("text"), str)
            else json.dumps(final_payload, ensure_ascii=False, indent=2)
        )

        final_summary = (
            f"User request:\n{redacted_prompt}\n\n"
            f"Step results:\n{payload_text}\n\n"
            "Write a concise, human-readable answer to the user's request based on the step results above.\n"
            "Rules:\n"
            "- Placeholders like $[EMAIL_1] or $[AMOUNT_1] represent real values. Write them exactly as-is in your response — they will be substituted before the user sees the output.\n"
            "- If the results indicate an error, no record found, or the action could not be completed, say so clearly and naturally.\n"
            "- Summarize what was accomplished or found. Do not dump raw JSON or repeat raw field names.\n"
            "- Do not invent values. Only use what appears in the step results."
        )

        _write_context_log(
            {
                "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
                "llm_instance": "finalizer",
                "model": self.openai_model,
                "run_id": self._run_id,
                "context": {
                    "system": self.system_prompt,
                    "user": final_summary,
                },
                "estimated_chars": len(self.system_prompt) + len(final_summary),
            }
        )

        resp = await self._openai.responses.create(
            model=self.openai_model,
            instructions=self.system_prompt,
            input=[{"role": "user", "content": final_summary}],
        )

        u = getattr(resp, "usage", None)
        if u:
            usage_stats["prompt_tokens"] += (
                getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0) or 0
            )
            usage_stats["completion_tokens"] += (
                getattr(u, "completion_tokens", None)
                or getattr(u, "output_tokens", 0)
                or 0
            )
            usage_stats["total_tokens"] += getattr(u, "total_tokens", 0) or 0

        final_text = (getattr(resp, "output_text", "") or "").strip()
        resolved = final_text
        for key, value in secrets.items():
            resolved = resolved.replace(f"$[{key}]", value)
        return resolved

    async def _run_planned_steps(
        self,
        *,
        redacted_prompt: str,
        planning: dict[str, Any],
        allowed_tools: Iterable[str],
        secrets: dict[str, str],
        usage_stats: dict[str, int],
        llm_call_count: int,
    ) -> tuple[str, dict[str, Any], int]:
        trace: list[dict[str, Any]] = []
        previous_results: dict[str, Any] = {}

        steps = _as_steps_list(planning)

        for idx, step in enumerate(steps):
            step_id = step.get("id")
            tool_name = self._resolve_chain_tool(step.get("tool_hint") or "")
            task = step.get("task")

            if not (
                isinstance(step_id, str)
                and isinstance(tool_name, str)
                and isinstance(task, str)
            ):
                continue

            tool_info = self._catalog.get(tool_name)
            if not tool_info:
                self._record_step_error(
                    trace,
                    previous_results,
                    step_id=step_id,
                    tool_name=tool_name,
                    error="Tool not found in catalog",
                )
                continue

            tool_def = _build_tool_def(tool_info)

            llm_call_count += 1
            logger.debug(
                f"LLM Call #{llm_call_count}: Execute planned step {step_id} via {tool_name}"
            )

            prev_for_step = self._prev_for_step(step, previous_results)
            secrets_for_step, visible_for_step = self._visible_for_step(
                task=task, step=step, prev_for_step=prev_for_step, secrets=secrets
            )

            prefilled = self._prefilled_args_from_step(step, visible_for_step=visible_for_step)
            structural_derived = self._prefill_derived_args(step, prev_for_step)
            prefilled = {**prefilled, **structural_derived}
            augmented_task = self._augment_task_with_derived_args(
                task, step, already_prefilled=set(structural_derived.keys())
            )

            # ablation: no_opaque_values, expose real secret values directly in the
            # executor context so the executor LLM sees raw sensitive data instead of placeholders.
            exec_task = augmented_task
            exec_prev = prev_for_step
            exec_visible = visible_for_step
            if "no_opaque_values" in self.ablations and secrets_for_step:
                real_vals = "\n".join(f"  {k}: {v}" for k, v in secrets_for_step.items())
                exec_task = augmented_task + f"\n\nContext values (use directly):\n{real_vals}"
                exec_prev = json.loads(_resolve_placeholders(json.dumps(prev_for_step, ensure_ascii=False), secrets)) if prev_for_step else prev_for_step
                exec_visible = set()

            parsed_args = await self._get_tool_args_from_llm(
                idx=idx,
                task=exec_task,
                tool_def=tool_def,
                tool_name=tool_name,
                visible_for_step=exec_visible,
                prev_for_step=exec_prev,
                usage_stats=usage_stats,
            )
            for k, v in prefilled.items():
                if isinstance(v, str) and _PLACEHOLDER_RE.fullmatch(v.strip()) and k in parsed_args:
                    pass  
                else:
                    parsed_args[k] = v

            (
                resolved,
                canon,
                resolved_pre_coerce,
                err,
            ) = await self._validate_and_resolve_args(
                parsed_args=parsed_args,
                tool_parameters=tool_info.parameters,
                secrets_for_step=secrets_for_step,
                visible_for_step=visible_for_step,
                secrets_all=secrets,
            )
            if err is not None:
                self._record_step_error(
                    trace,
                    previous_results,
                    step_id=step_id,
                    tool_name=tool_name,
                    error=err,
                )
                continue

            _PURE_PLACEHOLDER_RE = re.compile(r'^\$\[[\w]+\]$')
            for arg_name, pre_val in parsed_args.items():
                if isinstance(pre_val, str) and _PURE_PLACEHOLDER_RE.match(pre_val) and arg_name in resolved:
                    resolved[arg_name] = normalize_internal_id(resolved[arg_name])

            applied_rules = _compute_rules_applied(
                tool_def=tool_def,
                secrets_for_step=secrets_for_step,
                prev_for_step=prev_for_step,
                previous_results=previous_results,
                parsed_args_before=parsed_args,
                parsed_args_after=canon,
                resolved_args_before=resolved_pre_coerce,
                resolved_args_after=resolved,
                unresolved=[],
                concat_placeholders=False,
                format_errs=[],
            )

            logger.debug(
                f"[{step_id}] Calling tool={tool_name!r} server={tool_info.server!r}\n"
                f"  resolved_args={json.dumps(resolved, ensure_ascii=False)}\n"
                f"  secrets_for_step={json.dumps(secrets_for_step, ensure_ascii=False)}"
            )

            mcp_tool_name = tool_name.split(".")[-1] if "." in tool_name else tool_name
            tool_out = await self._call_tool(
                server=tool_info.server,
                tool_name=mcp_tool_name,
                resolved_args=resolved,
            )

            logger.debug(
                f"[{step_id}] Tool output ({tool_name!r}):\n"
                f"  {json.dumps(tool_out, ensure_ascii=False)}"
            )

            if "no_output_redaction" in self.ablations:
                from shardguard.core.models import OpaqueValues
                so = OpaqueValues(redacted=tool_out, secrets=secrets)
            else:
                so = await self._redactor.redact_obj(tool_out, secrets, usage_stats=usage_stats)

            self._audit(
                phase="executor_llm_call",
                step_id=step_id,
                tool=tool_name,
                recipient="executor_llm",
                event={
                    "data_items": {
                        "visible_placeholders": sorted(visible_for_step),
                        "visible_prev_steps": sorted(prev_for_step.keys()),
                    },
                    "information_type": "opaque_placeholders_and_redacted_results",
                    "rules_applied": applied_rules,
                    "privacy_properties": [
                        "isolation",
                        "least_privilege",
                        "policy_enforcement",
                    ],
                },
            )

            new_secret_keys = sorted(set(so.secrets) - set(secrets))
            secrets.update(so.secrets)

            _, executor_user_prompt = _build_executor_prompt(
                tool_name=tool_name,
                step_text=exec_task,
                visible_placeholders=exec_visible,
                previous_results=exec_prev,
            )
            trace.append(
                {
                    "step": step_id,
                    "tool": tool_name,
                    "server": tool_info.server,
                    "arguments": parsed_args,
                    "raw_result": tool_out,
                    "result": so.redacted,
                    "new_secrets_from_result": new_secret_keys,
                    "llm_context": executor_user_prompt,
                    "secrets_for_step": sorted(secrets_for_step.keys()),
                }
            )
            previous_results[step_id] = so.redacted

        llm_call_count += 1
        last_step_id = steps[-1].get("id") if steps else None
        final_payload = previous_results.get(last_step_id, previous_results)

        final_text = await self._finalize_answer(
            redacted_prompt=redacted_prompt,
            final_payload=final_payload,
            secrets=secrets,
            usage_stats=usage_stats,
        )

        return final_text, trace, llm_call_count

    # Oracle for debugging
    @staticmethod
    def _validate_oracle_plan(steps: list[dict[str, Any]], secrets: dict[str, str]) -> list[str]:
        bad: list[str] = []
        for step in steps:
            step_id = step.get("id", "<unknown>")
            for s in _iter_strings(step.get("args") or {}):
                for key in _extract_placeholder_keys(s):
                    if key not in secrets:
                        bad.append(f"{step_id}.args.{key}")
            for key in (step.get("placeholder_args") or {}).keys():
                if key not in secrets:
                    bad.append(f"{step_id}.placeholder_args.{key}")
            for key in _as_str_list(step.get("allowed_placeholders", [])):
                if key not in secrets:
                    bad.append(f"{step_id}.allowed_placeholders.{key}")
        return bad

    async def runJob(
        self,
        user_prompt: str,
        execute: bool,
        allowed_tools: Iterable[str] | None = None,
        oracle_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self._ensure_catalog()

        # 1. The uuid for this Loop
        self._run_id = str(uuid.uuid4())
        self._prompt_id = str(uuid.uuid4())
        llm_call_count = 0

        # 2. log the "job start" Event
        # This groups all subsequent events under this run_id
        self._audit(
            phase="job_start",
            step_id=None,
            tool=None,
            recipient="system",
            event={
                "message": "Coordination loop started",
                "execute_flag": execute,
                "input_length": len(user_prompt),
                "oracle_mode": oracle_plan is not None,
            },
        )

        usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        s0 = await self._redactor.redact_text(_normalize_compound_tokens(user_prompt), usage_stats=usage_stats)
        redacted_prompt = s0.redacted
        secrets: dict[str, str] = dict(s0.secrets)

        planning: dict[str, Any] | None = None

        if oracle_plan is not None:
            # Oracle for debugging
            resolved_steps = []
            for step in oracle_plan.get("steps", []):
                hint = step.get("tool_hint", "")
                resolved_hint = self._resolve_chain_tool(hint)
                step_id = step.get("id", "?")
                short_name = resolved_hint.split(".")[-1] if "." in resolved_hint else resolved_hint
                minimal_task = f"Call {short_name} for oracle step {step_id}. Use provided args, derived_args hints, and visible dependency outputs only."
                deps = list(step.get("depends_on") or [])
                for spec in (step.get("derived_args") or {}).values():
                    if isinstance(spec, dict):
                        from_step = spec.get("from_step")
                        if from_step and from_step not in deps:
                            deps.append(from_step)
                resolved_steps.append({**step, "tool_hint": resolved_hint, "task": minimal_task, "depends_on": deps})

            resolved_allowed = [
                self._resolve_chain_tool(t) for t in (oracle_plan.get("allowed_tools") or [])
            ]
            planning = {**oracle_plan, "steps": resolved_steps, "allowed_tools": resolved_allowed}
            allowed_tools = resolved_allowed

            bad_keys = self._validate_oracle_plan(resolved_steps, secrets)
            if bad_keys:
                return {
                    "run_id": self._run_id,
                    "error": f"Oracle plan rejected — unresolvable placeholders: {bad_keys}",
                    "secrets": secrets,
                    "trace": [],
                }
        elif allowed_tools is None:
            llm_call_count += 1
            logger.debug(f"LLM Call #{llm_call_count}: Planning")
            planning = await self._planner.plan(redacted_prompt, self._tool_summaries(), usage_stats=usage_stats)
            allowed_tools = planning.get("allowed_tools") or []

            known_tools: set[str] = set()
            for t in self._catalog._tools.values():
                known_tools.add(t.name)
                known_tools.add(f"{t.server}.{t.name}")
            valid_steps = []
            for step in planning.get("steps") or []:
                hint = step.get("tool_hint") or ""
                short = hint.split(".", 1)[-1] if "." in hint else hint
                if hint in known_tools or short in known_tools:
                    valid_steps.append(step)
                else:
                    logger.warning("Planner hallucinated tool %r — skipping step %s", hint, step.get("id"))
            planning["steps"] = valid_steps

        self._log_step_context(secrets, {}, f"User Request: {redacted_prompt}")

        if execute is False:
            return {
                "run_id": self._run_id,  
                "planning": planning,
            }

        if (
            planning is not None
            and isinstance(planning.get("steps"), list)
            and len(planning["steps"]) > 0
            and isinstance(planning["steps"][0], dict)
        ):
            final_text, step_trace, llm_call_count = await self._run_planned_steps(
                redacted_prompt=redacted_prompt,
                planning=planning,
                allowed_tools=allowed_tools or [],
                secrets=secrets,
                usage_stats=usage_stats,
                llm_call_count=llm_call_count,
            )

            return {
                "run_id": self._run_id,  
                "final_text": final_text,
                "trace": step_trace,
                "secrets": secrets,
                "secrets_keys": sorted(secrets.keys()),
                "usage": usage_stats,
                "planning": planning,
            }

        return {"run_id": self._run_id, "error": "No plan generated"}
