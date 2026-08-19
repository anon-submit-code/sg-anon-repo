import json
import logging
from datetime import UTC, datetime
from typing import Any

from openai import AsyncOpenAI  # type: ignore

from shardguard.core.prompts import PLANNING_PROMPT_FULL
from shardguard.core.tool_args_llm import _write_context_log

logger = logging.getLogger(__name__)


class PlanningLLM:
    """
    Least-privilege planner:
    - No access to tools
    - No access to the opaque-store keys list (only sees the redacted prompt)
    - Outputs a tool list
    """

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self.model = model

    @staticmethod
    def _extract_json_object(s: str) -> str | None:
        s = s.strip()
        if not s:
            return None
        if s.startswith("{") and s.endswith("}"):
            return s
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return s[start : end + 1]
        return None

    async def plan(
        self,
        redacted_prompt: str,
        tool_summaries: list[tuple[str, str, list[str], list[str]]],
        usage_stats: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        def _fmt(entry: tuple[str, str, list[str], list[str]]) -> str:
            name, desc, req, opt = entry
            req_str = f" [required: {', '.join(req)}]" if req else ""
            opt_str = f" [optional: {', '.join(opt)}]" if opt else ""
            return f"- {name}{req_str}{opt_str}: {desc}"

        tools_txt = "\n".join(_fmt(t) for t in tool_summaries)

        instructions = PLANNING_PROMPT_FULL
        user_msg = f"User request:\n{redacted_prompt}\n\nAvailable tools:\n{tools_txt}"

        _write_context_log(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "llm_instance": "planner",
                "model": self.model,
                "context": {
                    "system": instructions,
                    "user": user_msg,
                },
                "estimated_chars": len(instructions) + len(user_msg),
            }
        )

        resp = await self._client.responses.create(
            model=self.model,
            instructions=instructions,
            input=[{"role": "user", "content": user_msg}],
        )

        if usage_stats is not None:
            u = getattr(resp, "usage", None)
            if u:
                usage_stats["prompt_tokens"] += getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0) or 0
                usage_stats["completion_tokens"] += getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", 0) or 0
                usage_stats["total_tokens"] += getattr(u, "total_tokens", 0) or 0

        txt = (getattr(resp, "output_text", "") or "").strip()
        blob = self._extract_json_object(txt) or "{}"

        try:
            obj = json.loads(blob)
        except Exception as exc:
            logger.warning("Planner JSON parse failure: %s | raw=%r", exc, txt[:200])
            obj = {}

        allowed = obj.get("allowed_tools")
        steps = obj.get("steps")

        if not isinstance(allowed, list) or not all(
            isinstance(x, str) for x in allowed
        ):
            allowed = []

        steps2: list[dict[str, Any]] = []
        if isinstance(steps, list):
            for s in steps:
                if not isinstance(s, dict):
                    continue
                sid = s.get("id")
                task = s.get("task")
                tool_hint = s.get("tool_hint")
                depends_on = s.get("depends_on", [])
                placeholder_args = s.get("placeholder_args") or {}

                if not isinstance(sid, str) or not isinstance(task, str):
                    continue
                if tool_hint is not None and not isinstance(tool_hint, str):
                    tool_hint = None
                if not isinstance(depends_on, list) or not all(
                    isinstance(x, str) for x in depends_on
                ):
                    depends_on = []
                if not isinstance(placeholder_args, dict):
                    placeholder_args = {}

                args = s.get("args") or {}
                derived_args = s.get("derived_args") or {}
                if not isinstance(args, dict):
                    args = {}
                if not isinstance(derived_args, dict):
                    derived_args = {}

                steps2.append(
                    {
                        "id": sid,
                        "task": task,
                        "tool_hint": tool_hint,
                        "depends_on": depends_on,
                        "args": args,
                        "derived_args": derived_args,
                        "placeholder_args": placeholder_args,
                    }
                )
        else:
            steps2 = []

        seen: set[str] = set()
        allowed2: list[str] = []
        for name in allowed:
            if name not in seen:
                seen.add(name)
                allowed2.append(name)

        return {"allowed_tools": allowed2, "steps": steps2}
