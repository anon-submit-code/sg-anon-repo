# src/shardguard/registry.py
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from shardguard.core.mcp_client import MCPClient

logger = logging.getLogger(__name__)

REG_MCP_KEY = "mcps"

DEFAULT_HTTP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-06-18",
}

DEFAULT_TOOLS_CONFIG = {
    "allow": ["*"],
    "deny": [],
    "rename": {},
}


def load_registry(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {REG_MCP_KEY: {}}

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if REG_MCP_KEY not in data or not isinstance(data[REG_MCP_KEY], dict):
        data[REG_MCP_KEY] = {}
    return data


def _atomic_write(path: str, data: dict[str, Any]) -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + f".tmp.{int(time.time() * 1000)}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    shutil.move(str(tmp), str(p))


def add_mcp(
    registry_path: str,
    *,
    name: str,
    transport: str,
    description: str | None = None,
    http: dict[str, Any] | None = None,
    stdio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reg = load_registry(registry_path)
    mcps = reg[REG_MCP_KEY]

    tnorm = _normalize_transport(transport)
    if tnorm not in {"streamable-http", "stdio"}:
        raise ValueError("transport must be one of: streamable-http, stdio")

    entry: dict[str, Any] = {
        "transport": tnorm,
        "tools": DEFAULT_TOOLS_CONFIG.copy(),
    }
    if description:
        entry["description"] = description

    if tnorm == "streamable-http":
        entry["http"] = _build_http_entry(http)
    else:
        entry["stdio"] = _build_stdio_entry(stdio)

    mcps[name] = entry

    _atomic_write(registry_path, reg)
    return reg


def remove_mcp(registry_path: str, names: Iterable[str]) -> tuple[list[str], list[str]]:
    reg = load_registry(registry_path)
    mcps = reg[REG_MCP_KEY]
    removed, missing = [], []
    for n in names:
        if n in mcps:
            mcps.pop(n, None)
            removed.append(n)
        else:
            missing.append(n)
    _atomic_write(registry_path, reg)
    return removed, missing


def _parse_json_arg(arg: str | None) -> Any | None:
    if not arg:
        return None
    try:
        return json.loads(arg)
    except Exception:
        return None


def parse_transport_config(
    transport: str,
    url: str | None,
    headers: str | None,
    cmd: str | None,
    args: str | None,
    cwd: str | None,
    env: str | None,
    framing: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if transport in {"streamable-http", "http"}:
        if not url:
            raise ValueError("--url required for http")

        http_config = {"url": url}

        parsed_headers = _parse_json_arg(headers)
        if parsed_headers:
            http_config["headers"] = parsed_headers

        return http_config, None

    if not cmd:
        raise ValueError("--cmd required for stdio")

    stdio_config = {"cmd": cmd, "framing": framing}

    parsed_args = _parse_json_arg(args)
    if args is not None:
        stdio_config["args"] = parsed_args if parsed_args is not None else args

    if cwd:
        stdio_config["cwd"] = cwd

    parsed_env = _parse_json_arg(env)
    if parsed_env:
        stdio_config["env"] = parsed_env

    return None, stdio_config


def _normalize_transport(transport: str) -> str:
    t = (transport or "").strip().lower()
    if t in {"http", "streaming-http", "stream-http", "streamablehttp"}:
        return "streamable-http"
    return t


def _build_http_entry(http: dict[str, Any] | None) -> dict[str, Any]:
    if not http or not isinstance(http, dict) or not http.get("url"):
        raise ValueError("http.url is required for streamable-http")

    user_headers = http.get("headers") or {}
    combined_headers = DEFAULT_HTTP_HEADERS.copy()
    combined_headers.update(user_headers)

    return {
        "url": str(http["url"]).rstrip("/"),
        "headers": combined_headers,
    }


def _coerce_args_list(raw_args: Any) -> list[str]:
    if raw_args is None:
        return []
    if isinstance(raw_args, list) and all(isinstance(x, str) for x in raw_args):
        return raw_args
    if isinstance(raw_args, str):
        return [raw_args]
    raise ValueError("stdio.args must be a string or list[str]")


def _build_stdio_entry(stdio: dict[str, Any] | None) -> dict[str, Any]:
    if not stdio or not isinstance(stdio, dict) or not stdio.get("cmd"):
        raise ValueError("stdio.cmd is required for stdio")

    entry_stdio: dict[str, Any] = {
        "cmd": stdio["cmd"],
        "framing": (stdio.get("framing") or "jsonl").lower(),
        "args": _coerce_args_list(stdio.get("args")),
    }

    if stdio.get("cwd"):
        entry_stdio["cwd"] = stdio["cwd"]
    if stdio.get("env"):
        entry_stdio["env"] = stdio["env"]

    return entry_stdio


def _build_client_from_entry(name: str, cfg: dict[str, Any]) -> MCPClient:
    transport = (cfg.get("transport") or "").strip().lower()
    if transport in {"http", "streaming-http", "stream-http", "streamablehttp"}:
        transport = "streamable-http"

    if transport == "streamable-http":
        http = cfg.get("http") or {}
        url = str(http.get("url") or "").rstrip("/")
        headers = http.get("headers") or {}
        if not url:
            raise ValueError(f"{name!r}: http.url missing")
        return MCPClient(
            "streamable-http",
            http_url=url,
            http_headers=headers,
            session_id=f"sg-{name}",
        )

    if transport == "stdio":
        std = cfg.get("stdio") or {}
        cmd = std.get("cmd")
        if not cmd:
            raise ValueError(f"{name!r}: stdio.cmd missing")
        return MCPClient(
            "stdio",
            stdio_cmd=cmd,
            stdio_args=std.get("args") or [],
            stdio_cwd=std.get("cwd"),
            stdio_env=std.get("env") or {},
            session_id=f"sg-{name}",
            stdio_framing=(std.get("framing") or "jsonl").lower(),
        )

    raise ValueError(f"{name!r}: unsupported or missing transport")


_CLIENTS: dict[str, MCPClient] = {}
_STDIO_LOCKS: dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _get_stdio_lock(name: str) -> threading.Lock:
    with _REGISTRY_LOCK:
        if name not in _STDIO_LOCKS:
            _STDIO_LOCKS[name] = threading.Lock()
        return _STDIO_LOCKS[name]


def is_stdio_server(registry_path: str, name: str) -> bool:
    try:
        reg = load_registry(registry_path)
        cfg = reg.get(REG_MCP_KEY, {}).get(name) or {}
        return (cfg.get("transport") or "").strip().lower() == "stdio"
    except Exception:
        return False


def invalidate_client(registry_path: str, name: str) -> None:
    """Remove a cached client so the next call rebuilds it (forces reconnect)."""
    _CLIENTS.pop(name, None)


def get_or_create_client(registry_path: str, name: str) -> MCPClient:
    if name in _CLIENTS:
        return _CLIENTS[name]

    reg = load_registry(registry_path)
    mcps = reg.get(REG_MCP_KEY, {})
    cfg = mcps.get(name)
    if not cfg:
        raise KeyError(f"MCP {name!r} not found in registry")

    client = _build_client_from_entry(name, cfg)
    _CLIENTS[name] = client
    return client


def fetch_tools(
    registry_path: str,
    name: str,
    *,
    init: bool = True,
    timeout: float = 15.0,
) -> list[dict[str, Any]]:
    for attempt in range(2):
        client = get_or_create_client(registry_path, name)
        try:
            if init:
                client.initialize(timeout=timeout)
            return client.tools_list(timeout=timeout)
        except Exception as exc:
            if attempt == 0:
                logger.warning("fetch_tools %r failed (%s), reconnecting…", name, exc)
                invalidate_client(registry_path, name)
            else:
                raise


def fetch_all_tools(
    registry_path: str,
    *,
    init: bool = True,
    timeout: float = 15.0,
) -> dict[str, list[dict[str, Any]]]:
    reg = load_registry(registry_path)
    mcps = reg.get(REG_MCP_KEY, {})

    out: dict[str, list[dict[str, Any]]] = {}
    for name in mcps.keys():
        try:
            out[name] = fetch_tools(registry_path, name, init=init, timeout=timeout)
        except Exception as exc:
            logger.warning("Failed to fetch tools from %r: %s", name, exc)
            out[name] = []
    return out


def clear_client_cache() -> None:
    global _CLIENTS  # noqa: PLW0602
    for c in _CLIENTS.values():
        try:
            c.close()
        except Exception as exc:
            logger.warning("Failed to close MCP client: %s", exc)
    _CLIENTS.clear()
