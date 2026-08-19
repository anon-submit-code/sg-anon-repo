import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PLACEHOLDER_RE = re.compile(r"\$\[([A-Za-z0-9_]+)\]")  # preferred: $[KEY]
_DOLLAR_KEY_RE = re.compile(
    r"\$([A-Za-z][A-Za-z0-9_]*)\$?"
)  # tolerated: $KEY or $KEY$ (could happen in LLM provider mistakes)
_P_TOKEN_RE = re.compile(r"\[\[P(\d+)\]\]")  # tolerated: [[Pn]] (redactor format, LLM may emit in tool args)


def _is_valid_email(s: str) -> bool:
    return bool(_EMAIL_RE.match(s.strip()))


def _is_valid_datetime(s: str) -> bool:
    try:
        t = s.strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"

        datetime.fromisoformat(t)
        return True
    except Exception:
        return False


_PLACEHOLDER_TOKEN_RE = re.compile(r"^\$\[[A-Za-z0-9_]+\]$")


def _canonical_placeholder_token(key: str) -> str:
    return f"$[{key}]"


def _extract_placeholder_keys(s: str) -> list[str]:
    keys: list[str] = []
    for m in _PLACEHOLDER_RE.finditer(s):
        keys.append(m.group(1))
    for m in _DOLLAR_KEY_RE.finditer(s):
        keys.append(m.group(1))
    return keys


def _is_scalar_placeholder_allowed(prop_schema: dict[str, Any]) -> bool:
    # For scalar type fields where a placeholder should stand in for the whole value
    t = prop_schema.get("type")
    fmt = prop_schema.get("format")
    if fmt in ("email", "date-time"):
        return True
    if t in ("integer", "number", "boolean"):
        return True
    return False


def _canonicalize_scalar_placeholders(
    args: dict[str, Any], schema: dict[str, Any], available_keys: set[str]
) -> dict[str, Any]:

    if not isinstance(args, dict) or not isinstance(schema, dict):
        return args

    out: dict[str, Any] = dict(args)
    props = schema.get("properties")
    if not isinstance(props, dict):
        return out

    for k, v in list(out.items()):
        prop_schema = props.get(k) if isinstance(props.get(k), dict) else None
        if not prop_schema or not isinstance(v, str):
            continue
        if not _is_scalar_placeholder_allowed(prop_schema):
            continue

        keys = [kk for kk in _extract_placeholder_keys(v) if kk in available_keys]
        uniq = []
        for kk in keys:
            if kk not in uniq:
                uniq.append(kk)
        if len(uniq) == 1 and v.strip() != _canonical_placeholder_token(uniq[0]):
            # If there is other text around, collapse to placeholder only.
            out[k] = _canonical_placeholder_token(uniq[0])

    return out


def _iter_option_schemas(prop_schema: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("anyOf", "oneOf"):
        options = prop_schema.get(key)
        if isinstance(options, list):
            return [s for s in options if isinstance(s, dict)]
    return [prop_schema]


def _add_format_error(
    errs: list[dict[str, Any]],
    *,
    key: str,
    validation: str,
    message: str,
) -> None:
    errs.append(
        {
            "path": [key],
            "code": "invalid_string",
            "validation": validation,
            "message": message,
        }
    )


def _as_str_list(v: Any) -> list[str]:
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return v
    return []


def _as_steps_list(planning: dict[str, Any]) -> list[dict[str, Any]]:
    steps = planning.get("steps") or []
    if not isinstance(steps, list):
        return []
    return [s for s in steps if isinstance(s, dict)]


def _is_allowed_tool(tool_name: str, allowed_set: set[str]) -> bool:
    return (not allowed_set) or (tool_name in allowed_set)


def _build_tool_def(tool_info: Any) -> dict[str, Any]:
    # OpenAI requires ^[a-zA-Z0-9_-]+$ — strip server prefix if present
    name = tool_info.name
    if "." in name:
        name = name.split(".")[-1]
    return {
        "type": "function",
        "name": name,
        "description": tool_info.description,
        "parameters": tool_info.parameters,
        "strict": True,
    }


def _validate_formats(
    args: dict[str, Any], schema: dict[str, Any]
) -> list[dict[str, Any]]:
    errs: list[dict[str, Any]] = []

    if not isinstance(args, dict) or not isinstance(schema, dict):
        return errs

    props = schema.get("properties")
    if not isinstance(props, dict):
        return errs

    validators: dict[str, tuple[Callable[[str], bool], str, str]] = {
        "email": (_is_valid_email, "email", "Invalid email"),
        "date-time": (_is_valid_datetime, "datetime", "Invalid datetime"),
    }

    for key, prop_schema_any in props.items():
        if key not in args or not isinstance(prop_schema_any, dict):
            continue

        value = args.get(key)
        if not isinstance(value, str):
            continue

        for opt_schema in _iter_option_schemas(prop_schema_any):
            fmt = opt_schema.get("format")
            entry = validators.get(fmt) if isinstance(fmt, str) else None
            if entry is None:
                continue

            is_valid, validation, message = entry
            if not is_valid(value):
                _add_format_error(errs, key=key, validation=validation, message=message)
            break  # stop after first recognized format among options

    return errs


def _to_number(v: Any) -> float:
    if isinstance(v, str):
        return float(v.strip().lstrip("$").replace(",", ""))
    return float(v)


def _to_integer(v: Any) -> int:
    if isinstance(v, str):
        return int(v.strip().lstrip("$").replace(",", ""))
    return int(v)


_TYPE_CASTERS: dict[str, Any] = {
    "string": str,
    "number": _to_number,
    "integer": _to_integer,
    "boolean": lambda v: v if isinstance(v, bool) else str(v).lower() == "true",
    "array": lambda v: v
    if isinstance(v, list)
    else ([v] if v not in (None, "") else []),
    "object": lambda v: v if isinstance(v, dict) else {},
}

_PYTHON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_type(spec: dict[str, Any]) -> str | None:
    if "type" in spec:
        return spec["type"]
    for sub in spec.get("anyOf", []):
        if isinstance(sub, dict) and sub.get("type") not in (None, "null"):
            return sub["type"]
    return None


def _coerce_types(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    props = (schema or {}).get("properties") or {}
    out = dict(args)
    for k, spec in props.items():
        if k not in out:
            continue
        expected = _schema_type(spec)
        if not expected:
            continue
        python_type = _PYTHON_TYPES.get(expected)
        caster = _TYPE_CASTERS.get(expected)
        if caster and python_type and not isinstance(out[k], python_type):
            try:
                out[k] = caster(out[k])
            except Exception:
                pass
    return out


# Matches full URLs to protect their internal slashes from splitting
_URL_PROTECT_RE = re.compile(r"https?://\S+", re.IGNORECASE)
# Matches standalone word/word tokens (bounded by whitespace or string boundary)
_SLASH_TOKEN_RE = re.compile(r"(?<!\S)([A-Za-z][A-Za-z0-9_-]*)\/([A-Za-z][A-Za-z0-9_-]*)(?!\S)")


def _normalize_compound_tokens(text: str) -> str:
    protected: list[str] = []

    def _stash(m: re.Match) -> str:
        idx = len(protected)
        protected.append(m.group(0))
        return f"\x00URL{idx}\x00"

    text = _URL_PROTECT_RE.sub(_stash, text)
    text = _SLASH_TOKEN_RE.sub(r"\1 \2", text)
    for i, original in enumerate(protected):
        text = text.replace(f"\x00URL{i}\x00", original)
    return text


class SanitizationResult:
    def __init__(
        self, sanitized_input: str, changes_made: list[str], original_length: int
    ):
        self.sanitized_input = sanitized_input
        self.changes_made = changes_made
        self.original_length = original_length
        self.final_length = len(sanitized_input)


class InputSanitizer:
    def __init__(self, console: Console | None = None, max_length: int = 10000):
        self.console = console or Console()
        self.max_length = max_length
        self.dangerous_patterns = [
            (r"<script[^>]*>.*?</script>", "Script tags"),
            (r"javascript:", "JavaScript URLs"),
            (r"data:text/html", "HTML data URLs"),
        ]

    def sanitize(
        self, user_input: str, show_progress: bool = True
    ) -> SanitizationResult:
        if show_progress:
            self._show_sanitization_start()
            self._show_original_input(user_input)

        if not user_input or not user_input.strip():
            if show_progress:
                self.console.print(
                    "Error: User input cannot be empty"
                )
            raise ValueError("User input cannot be empty")

        changes_made = []
        original_length = len(user_input)
        sanitized = user_input

        sanitized, step_changes = self._normalize_whitespace(sanitized)
        changes_made.extend(step_changes)

        sanitized, step_changes = self._remove_control_characters(sanitized)
        changes_made.extend(step_changes)

        sanitized, step_changes = self._truncate_long_input(sanitized)
        changes_made.extend(step_changes)

        sanitized, step_changes = self._remove_dangerous_patterns(sanitized)
        changes_made.extend(step_changes)

        result = SanitizationResult(sanitized, changes_made, original_length)

        if show_progress:
            self._show_sanitization_results(result, user_input)

        return result

    def _normalize_whitespace(self, text: str) -> tuple[str, list[str]]:
        original = text.strip()
        normalized = re.sub(r"\s+", " ", original)

        changes = []
        if normalized != original:
            changes.append(" Normalized whitespace and line endings")

        return normalized, changes

    def _remove_control_characters(self, text: str) -> tuple[str, list[str]]:
        before_removal = text
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        changes = []
        if cleaned != before_removal:
            changes.append("Removed dangerous control characters")

        return cleaned, changes

    def _truncate_long_input(self, text: str) -> tuple[str, list[str]]:
        if len(text) > self.max_length:
            raise ValueError(
                f"Input truncated: exceeds maximum length of {self.max_length} characters"
            )

        return text, []

    def _remove_dangerous_patterns(self, text: str) -> tuple[str, list[str]]:
        changes = []

        for pattern, description in self.dangerous_patterns:
            before_removal = text
            text = re.sub(
                pattern, "", text, flags=re.IGNORECASE | re.DOTALL
            )
            if text != before_removal:
                changes.append(f"✓ Removed {description}")

        return text, changes

    def _show_sanitization_start(self):
        self.console.print("\n[bold blue]🔍 Input Sanitization Process[/bold blue]")

    def _show_original_input(self, user_input: str):
        display_input = user_input[:200] + ("..." if len(user_input) > 200 else "")
        original_panel = Panel(
            display_input,
            title="[bold]Original Input[/bold]",
            border_style="dim",
        )
        self.console.print(original_panel)

    def _show_sanitization_results(
        self, result: SanitizationResult, original_input: str
    ):
        """Display sanitization results and changes."""
        if result.changes_made:
            changes_text = Text()
            for change in result.changes_made:
                changes_text.append(change + "\n", style="green")

            changes_panel = Panel(
                changes_text,
                title="[bold green]Sanitization Changes[/bold green]",
                border_style="green",
            )
            self.console.print(changes_panel)
        else:
            self.console.print(
                "[green] No sanitization needed input is clean[/green]"
            )

        if result.sanitized_input != original_input:
            display_sanitized = result.sanitized_input[:200] + (
                "..." if len(result.sanitized_input) > 200 else ""
            )
            sanitized_panel = Panel(
                display_sanitized,
                title="[bold]Sanitized Input[/bold]",
                border_style="green",
            )
            self.console.print(sanitized_panel)

        if result.final_length != result.original_length:
            self.console.print(
                f"[dim]Length: {result.original_length} → {result.final_length} characters[/dim]"
            )

    def add_dangerous_pattern(self, pattern: str, description: str):
        self.dangerous_patterns.append((pattern, description))

    def remove_dangerous_pattern(self, pattern: str):
        self.dangerous_patterns = [
            (p, d) for p, d in self.dangerous_patterns if p != pattern
        ]
