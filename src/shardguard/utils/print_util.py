from __future__ import annotations

import json
from typing import Any

from rich.console import Console

console = Console()


def print_json(data: Any, pretty: bool = True) -> None:
    if pretty:
        console.print_json(data=data)
    else:
        print_json(json.dumps(data, separators=(",", ":"), ensure_ascii=False))


def log_success(msg: str):
    console.print(f"[green]{msg}[/green]")


def log_err(msg: str):
    console.print(f"[red]{msg}[/red]")
