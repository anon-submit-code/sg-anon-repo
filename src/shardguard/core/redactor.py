from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from shardguard.core.models import OpaqueValues
from shardguard.core.tool_args_llm import _write_context_log

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^https?://", re.I)
_PHONE_RE = re.compile(r"^\+?\d[\d\-\s\(\)]{7,14}\d$")
_EXISTING_PLACEHOLDER_RE = re.compile(r"\$\[[A-Za-z0-9_]+\]")
_LONG_NUMBER_RE = re.compile(r"\b\d{4,}\b")
_SPECIAL_CHAR_TOKEN_RE = re.compile(r"\S*[^a-zA-Z0-9\s]\S*")
_ALPHANUMERIC_CODE_RE = re.compile(r"\b(?:[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z]+[A-Za-z0-9]*)\b")

_FILE_PATH_RE = re.compile(r"^(/[^/\s]+)+/?$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.[a-zA-Z]{1,6}$")
_SNAKE_KEBAB_RE = re.compile(r"^[a-z][a-z0-9]*([_\-][a-z0-9]+)+$").
_JSON_KEY_RE = re.compile(r"""^['"]?[A-Za-z_][A-Za-z0-9_\- ]*['"]?\s*:$""")
_CONTRACTION_RE = re.compile(r"^\w+'\w+$")
_TITLE_ABBREV_RE = re.compile(r"^(Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|Rev|Gen|Sgt|Cpl|Pfc|Pvt|Lt|Cdr|Capt|Maj|Col|Brig|Adm|Gov|Sen|Rep|Atty|Hon|Msgr|Rt)\.?$", re.I)


def _looks_like_json_key_fragment(token: str) -> bool:
    s = token.strip()
    s = s.replace('\\"', '"').replace("\\'", "'")
    s = s.rstrip(",)}]").lstrip("{[(")
    return bool(_JSON_KEY_RE.fullmatch(s))

_DOLLAR_AMOUNT_RE = re.compile(r"^\$[\d,]+(\.\d+)?$")
_SIMPLE_DECIMAL_RE = re.compile(r"^\d+\.\d+$")

_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_DOB_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PASSPORT_RE = re.compile(r"^[A-Z]{2}-[A-Z]-[A-Z0-9]+$")
_CARD_RE = re.compile(r"^\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}$")
_GOV_ID_RE = re.compile(r"^[A-Z]{2}-[A-Z]{2}-[A-Z0-9]+$")
_PASSWORD_RE = re.compile(r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[^A-Za-z0-9\s])\S{8,}$")

_KNOWN_KINDS = {
    "EMAIL", "SSN", "DOB", "PASSPORT", "GOV_ID", "CARD", "ACCOUNT",
    "ROUTING", "PHONE", "PASSWORD", "ADDRESS", "URL", "SECRET",
}

_NAME_RE = re.compile(r"^[A-Z][a-z]+([\s\-][A-Z][a-z]+)+$")


def _classify_kind(value: str) -> str:
    v = (value or "").strip()
    if _EMAIL_RE.match(v):
        return "EMAIL"
    if _SSN_RE.match(v):
        return "SSN"
    if _DOB_RE.match(v):
        return "DOB"
    if _PASSPORT_RE.match(v):
        return "PASSPORT"
    if _GOV_ID_RE.match(v):
        return "GOV_ID"
    if _CARD_RE.match(v):
        return "CARD"
    if _PASSWORD_RE.match(v):
        return "PASSWORD"
    if _URL_RE.match(v):
        return "URL"
    if _PHONE_RE.match(v):
        return "PHONE"
    if _NAME_RE.match(v):
        return "NAME"
    return "SECRET"


def _extract_json_object(s: str) -> str | None:
    s = (s or "").strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return None


def _apply_redactions(
    text: str,
    sensitive_values: list[str] | list[tuple[str, str]],
    prior_secrets: dict[str, str],
) -> OpaqueValues:
    labeled: list[tuple[str, str]] = []
    for item in sensitive_values:
        if isinstance(item, tuple):
            labeled.append(item)
        else:
            labeled.append((item, _classify_kind(item)))
    for v in _LONG_NUMBER_RE.findall(text):
        labeled.append((v, _classify_kind(v)))
    for v in _ALPHANUMERIC_CODE_RE.findall(text):
        if 2 <= len(v) <= 15 and "." not in v:
            labeled.append((v, "SECRET"))
    for v in _SPECIAL_CHAR_TOKEN_RE.findall(text):
        if len(v) <= 1:
            continue 
        inner = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", v)
        if inner and not re.search(r"[^a-zA-Z0-9]", inner):
            continue
        if not _FILE_PATH_RE.match(v) and not _FILENAME_RE.match(v) and not _SNAKE_KEBAB_RE.match(v) and not _looks_like_json_key_fragment(v) and not _DOLLAR_AMOUNT_RE.match(v) and not _CONTRACTION_RE.match(v) and not _SIMPLE_DECIMAL_RE.match(v) and not _TITLE_ABBREV_RE.match(v):
            labeled.append((v, _classify_kind(v)))

    secrets = dict(prior_secrets)

    counters: dict[str, int] = {}
    for k in secrets:
        m = re.match(r"^([A-Z_]+)_(\d+)$", k)
        if m:
            kind, idx = m.group(1), int(m.group(2))
            counters[kind] = max(counters.get(kind, 0), idx)

    def new_key(kind: str) -> str:
        n = counters.get(kind, 0) + 1
        counters[kind] = n
        return f"{kind}_{n}"

    def first_pos(pair: tuple[str, str]) -> int:
        idx = text.find(pair[0])
        return idx if idx != -1 else len(text)

    seen_values = set(secrets.values())
    for value, kind in sorted(set(labeled), key=first_pos):
        if not value or value not in text:
            continue
        if _EXISTING_PLACEHOLDER_RE.fullmatch(value.strip()):
            continue
        if value not in seen_values:
            secrets[new_key(kind)] = value
            seen_values.add(value)

    value_to_ph = {v: f"$[{k}]" for k, v in secrets.items()}
    redacted = text
    for value in sorted(value_to_ph, key=len, reverse=True):
        if value in redacted:
            redacted = redacted.replace(value, value_to_ph[value])

    return OpaqueValues(redacted=redacted, secrets=secrets)


class LlmOpaqueRedactor:

    def __init__(
        self, client: AsyncOpenAI, *, model: str, prompt_template: str
    ) -> None:
        self._client = client
        self.model = model
        self.prompt_template = prompt_template

    def _format_prompt(self, user_input: str) -> str:
        return self.prompt_template.replace("{user_prompt}", user_input)

    async def _identify(self, text: str, usage_stats: dict[str, int] | None = None) -> list[tuple[str, str]]:
        from datetime import UTC, datetime
        prompt_content = self._format_prompt(text)
        _write_context_log({
            "timestamp": datetime.now(UTC).isoformat(),
            "llm_instance": "redactor",
            "model": self.model,
            "context": {"user": prompt_content},
            "estimated_chars": len(prompt_content),
        })
        resp = await self._client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt_content}],
            max_output_tokens=900,
        )
        if usage_stats is not None:
            u = getattr(resp, "usage", None)
            if u:
                usage_stats["prompt_tokens"] += getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0) or 0
                usage_stats["completion_tokens"] += getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", 0) or 0
                usage_stats["total_tokens"] += getattr(u, "total_tokens", 0) or 0
        raw = (getattr(resp, "output_text", "") or "").strip()
        blob = _extract_json_object(raw) or "{}"
        try:
            obj = json.loads(blob)
            items = obj.get("sensitive_values", [])
            result: list[tuple[str, str]] = []
            for item in items:
                if isinstance(item, dict):
                    v = item.get("value", "")
                    k = item.get("kind", "SECRET")
                    if isinstance(v, str) and v:
                        pattern_kind = _classify_kind(v)
                        effective_kind = pattern_kind if pattern_kind != "SECRET" else (k if k in _KNOWN_KINDS else "SECRET")
                        result.append((v, effective_kind))
                elif isinstance(item, str) and item:
                    result.append((item, _classify_kind(item)))
            return result
        except Exception as exc:
            logger.warning("Redactor JSON parse failure: %s | raw=%r", exc, raw[:200])
            return []

    async def redact_text(
        self, text: str, prior_secrets: dict[str, str] | None = None,
        usage_stats: dict[str, int] | None = None,
    ) -> OpaqueValues:
        sensitive = await self._identify(text, usage_stats=usage_stats)
        return _apply_redactions(text, sensitive, dict(prior_secrets or {}))

    _MAX_REDACT_STR_CHARS: int = 400

    async def redact_obj(
        self, obj: Any, prior_secrets: dict[str, str] | None = None,
        usage_stats: dict[str, int] | None = None,
    ) -> OpaqueValues:
        secrets = dict(prior_secrets or {})

        async def walk(x: Any) -> Any:
            nonlocal secrets
            if isinstance(x, str):
                if len(x) > self._MAX_REDACT_STR_CHARS:
                    try:
                        return await walk(json.loads(x))
                    except (json.JSONDecodeError, ValueError):
                        res = _apply_redactions(x, [], secrets)
                        secrets = res.secrets
                        return res.redacted
                stripped = x.strip()
                if stripped.startswith(("{", "[")):
                    try:
                        return await walk(json.loads(x))
                    except (json.JSONDecodeError, ValueError):
                        pass
                res = await self.redact_text(x, prior_secrets=secrets, usage_stats=usage_stats)
                secrets = res.secrets
                return res.redacted
            if isinstance(x, list):
                return [await walk(i) for i in x]
            if isinstance(x, dict):
                out: dict[str, Any] = {}
                for k, v in x.items():
                    out[k] = await walk(v)
                return out
            return x

        redacted = await walk(obj)
        return OpaqueValues(redacted=redacted, secrets=secrets)
