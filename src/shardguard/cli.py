"""ShardGuard CLI - Command-line interface for safe task execution."""

from __future__ import annotations

import asyncio
import json
import logging
import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shardguard.core.baseline import BaselineAgent
from shardguard.core.coordination import CoordinationService
from shardguard.mcp_servers import registry as reg_lib
from shardguard.utils.print_util import (
    log_err,
    log_success,
    print_json,
)

EXECUTOR_MODEL_OPTION = typer.Option(
    None,
    "--executor-model",
    help="Model for executor provider (defaults per provider).",
)


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

app = typer.Typer(
    help="ShardGuard CLI",
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=True,
)
console = Console()

registry_app = typer.Typer(no_args_is_help=True)
app.add_typer(registry_app, name="registry")


def _validate_gemini_api_key(provider: str, api_key: str | None) -> None:
    """Validate Gemini API key if required."""
    if provider == "gemini" and not api_key:
        console.print(
            "[bold red]Error:[/bold red] Gemini API key required. "
            "Set GEMINI_API_KEY env var or use --gemini-api-key"
        )
        raise typer.Exit(1)


def _get_model_for_provider(provider: str, model: str | None) -> str:
    """Get the model name for the provider, auto-detecting if not specified."""
    if model is not None:
        return model
    return "gemini-2.0-flash-exp" if provider == "gemini" else "llama3.2"


PROVIDER_OPTION = typer.Option(
    "ollama", "--provider", help="LLM provider (ollama or gemini)"
)
MODEL_OPTION = typer.Option(
    None, "--model", help="Model to use (auto-detected if not specified)"
)
OLLAMA_URL_OPTION = typer.Option(
    "http://localhost:11434", "--ollama-url", help="Ollama base URL"
)
GEMINI_API_KEY_OPTION = typer.Option(
    None, "--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY env var)"
)
VERBOSE_OPTION = typer.Option(
    False, "--verbose", "-v", help="Show detailed information"
)

REGISTRY_PATH_OPTION = typer.Option(
    "src/shardguard/mcp_servers/mcp_registry.json",
    "--registry-path",
    help="Path to the MCP registry JSON file",
)
OPENAI_MODEL_OPTION = typer.Option(
    "gpt-4o-mini",
    "--openai-model",
    help="OpenAI model name",
)


@app.command()
def list_tools(
    registry_path: str = REGISTRY_PATH_OPTION,
    verbose: bool = VERBOSE_OPTION,
):
    """List all available MCP tools."""

    async def _list_tools() -> None:
        try:
            tools_by_server = await asyncio.to_thread(
                reg_lib.fetch_all_tools,
                registry_path,
                init=True,
                timeout=15.0,
            )

            console.print("[bold blue]Available MCP Tools (by server):[/bold blue]\n")

            for server_name, tools in tools_by_server.items():
                console.print(f"[bold]{server_name}[/bold]  ({len(tools)} tools)")

                if not tools:
                    console.print("  [dim]— no tools or failed to fetch —[/dim]\n")
                    continue

                for t in tools:
                    tool_name = t.get("name", "")
                    title = t.get("title") or ""
                    desc = t.get("description") or ""
                    tool_key = f"{server_name}.{tool_name}"

                    if verbose:
                        console.print(f"  • [cyan]{tool_key}[/cyan]")
                        if title:
                            console.print(f"      [dim]title:[/dim] {title}")
                        if desc:
                            console.print(f"      [dim]desc:[/dim] {desc}")
                    else:
                        label = tool_key if not title else f"{tool_key} — {title}"
                        console.print(f"  • [cyan]{label}[/cyan]")

                console.print()

        finally:
            await asyncio.to_thread(reg_lib.clear_client_cache)

    asyncio.run(_list_tools())


@app.command("plan")
def plan(
    prompt: str = typer.Argument(..., help="User request"),
    execute: bool = typer.Option(
        False,
        "-x",
        help="Execute tool calling",
    ),
    provider: str = PROVIDER_OPTION,
    model: str = MODEL_OPTION,
    json_out: bool = typer.Option(False, "--json"),
    registry_path: str = REGISTRY_PATH_OPTION,
    openai_model: str = OPENAI_MODEL_OPTION,
    executor_model: str | None = EXECUTOR_MODEL_OPTION,
    api_key: str | None = GEMINI_API_KEY_OPTION,
):
    """Show the execution plan, pass -x to also run it."""
    async def _run():
        _validate_gemini_api_key(provider, api_key)
        detected_model = _get_model_for_provider(provider, model)
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[bold red]Error:[/bold red] OPENAI_API_KEY is not set. ")
            raise typer.Exit(1)
        console.print(f"[dim]OpenAI chain model: {openai_model}[/dim]")

        coord = CoordinationService(
            registry_path=registry_path,
            openai_model=openai_model,  # for Coordinator to chain subprompts
            step_executor=detected_model,  # "ollama" | "gemini" | "openai"
            ollama_model=(executor_model or "llama3.2"),
            gemini_model=(executor_model or "gemini-2.0-flash-exp"),
            gemini_api_key=api_key,
        )

        try:
            result = await coord.runJob(prompt, execute)
        except Exception as e:
            logging.getLogger(__name__).exception("coord.run failed")
            console.print(f"[bold red]Coordinator error:[/bold red] {e}")
            console.print("[dim]See shardguard_debug.log for details.[/dim]")
            raise typer.Exit(1)

        if json_out:
            print_json(result)
            return

        console.print(
            Panel(
                f"[bold]User Prompt:[/bold] {prompt}",
                title="",
                border_style="green",
            )
        )

        # Print trace of tool calls
        trace = result.get("trace") or []
        if trace:
            table = Table(title="", expand=True)
            table.add_column("Step", justify="right", style="cyan", no_wrap=True)
            table.add_column("Tool", style="magenta")
            table.add_column("Opaque values", style="yellow")
            table.add_column("Output (redacted)", style="white")

            for i, item in enumerate(trace, start=1):
                tool = str(item.get("tool", ""))
                args = item.get("arguments", {})
                out = (
                    item.get("result")
                    or item.get("result_handle")
                    or item.get("output")
                    or {}
                )

                table.add_row(
                    str(i),
                    tool,
                    json.dumps(args, ensure_ascii=False),
                    json.dumps(out, ensure_ascii=False),
                )
            console.print(table)

        final_text = result.get("final_text") or ""
        usage = result.get("usage") or {}
        if usage:
            console.print(
                Panel(
                    f"prompt_tokens={usage.get('prompt_tokens', 0)}\n"
                    f"completion_tokens={usage.get('completion_tokens', 0)}\n"
                    f"total_tokens={usage.get('total_tokens', 0)}",
                    title="Tokens used",
                    border_style="magenta",
                )
            )

        if final_text:
            console.print(Panel(final_text, title="Response", border_style="blue"))
        else:
            console.print("[dim](No execution response text returned)[/dim]")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.getLogger(__name__).exception("ShardGuard CLI crashed")
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command("baseline")
def baseline(
    prompt: str = typer.Argument(..., help="User request"),
    registry_path: str = REGISTRY_PATH_OPTION,
    openai_model: str = OPENAI_MODEL_OPTION,
    json_out: bool = typer.Option(False, "--json"),
):
    """For running without shardguard, has tools, but no redaction or planning."""

    async def _run():
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[bold red]Error:[/bold red] OPENAI_API_KEY is not set.")
            raise typer.Exit(1)
        console.print(f"[dim]Baseline model: {openai_model}[/dim]")
        console.print("[dim](No redaction, no planner, no per-step scoping)[/dim]")

        agent = BaselineAgent(registry_path=registry_path, openai_model=openai_model)
        try:
            result = await agent.run(prompt)
        except Exception as e:
            logging.getLogger(__name__).exception("baseline.run failed")
            console.print(f"[bold red]Baseline error:[/bold red] {e}")
            raise typer.Exit(1)

        if json_out:
            print_json(result)
            return

        console.print(
            Panel(
                f"[bold]User Prompt:[/bold] {prompt}",
                title="Baseline (no ShardGuard)",
                border_style="yellow",
            )
        )

        trace = result.get("trace") or []
        if trace:
            table = Table(title="", expand=True)
            table.add_column("Call", justify="right", style="cyan", no_wrap=True)
            table.add_column("Tool", style="magenta")
            table.add_column("Arguments", style="yellow")
            table.add_column("Output", style="white")

            for i, item in enumerate(trace, start=1):
                table.add_row(
                    str(i),
                    str(item.get("tool", "")),
                    json.dumps(item.get("arguments", {}), ensure_ascii=False),
                    json.dumps(item.get("result", {}), ensure_ascii=False),
                )
            console.print(table)

        usage = result.get("usage") or {}
        if usage:
            console.print(
                Panel(
                    f"prompt_tokens={usage.get('prompt_tokens', 0)}\n"
                    f"completion_tokens={usage.get('completion_tokens', 0)}\n"
                    f"total_tokens={usage.get('total_tokens', 0)}",
                    title="Tokens used",
                    border_style="magenta",
                )
            )

        final_text = result.get("final_text") or ""
        if final_text:
            console.print(Panel(final_text, title="Response", border_style="yellow"))
        else:
            console.print("[dim](No response text returned)[/dim]")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.getLogger(__name__).exception("ShardGuard CLI crashed")
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """ShardGuard CLI, Generate safe execution plans for user prompts."""
    if ctx.invoked_subcommand is None:

        async def _init():
            console.print("[bold blue]Welcome to ShardGuard![/bold blue]")
            console.print("\n[dim]Use --help to see available commands.[/dim]")
            console.print("[dim]Available commands: list-tools, plan, baseline[/dim]")

        asyncio.run(_init())


@registry_app.command("add-mcp")
def registry_add_mcp(
    registry: str = typer.Option(..., "--registry", "-r"),
    name: str = typer.Option(..., "--name", "-n"),
    transport: str = typer.Option(..., "--transport", "-t"),
    description: str | None = typer.Option(None, "--desc"),
    url: str | None = typer.Option(None, "--url"),
    headers: str | None = typer.Option(None, "--headers"),
    cmd: str | None = typer.Option(None, "--cmd"),
    args: str | None = typer.Option(None, "--args"),
    cwd: str | None = typer.Option(None, "--cwd"),
    env: str | None = typer.Option(None, "--env"),
    framing: str = typer.Option("jsonl", "--framing"),
):
    http_config = None
    stdio_config = None

    try:
        http_config, stdio_config = reg_lib.parse_transport_config(
            transport, url, headers, cmd, args, cwd, env, framing
        )
        reg_lib.add_mcp(
            registry_path=registry,
            name=name,
            transport=transport,
            description=description,
            http=http_config,
            stdio=stdio_config,
        )

        log_success(f"Added MCP {name}")
    except ValueError as e:
        log_err(str(e))
        raise typer.Exit(1)


@registry_app.command("remove-mcp")
def registry_rm_mcp(
    registry: str = typer.Option(..., "--registry", "-r"),
    names: list[str] = typer.Argument(...),
):
    try:
        removed, missing = reg_lib.remove_mcp(registry, names)

        if removed:
            log_success(f"Successfully removed: {', '.join(removed)}")
        if missing:
            log_err(f"Services not found: {', '.join(missing)}")

    except Exception as e:
        log_err(f"Failed to remove services: {e}")
        raise typer.Exit(1)


@app.command("eval")
def eval_cmd(
    category: str = typer.Option(None, "--category", "-c", help="Run only this category"),
    mode: str = typer.Option("both", "--mode", "-m", help="shardguard | baseline | both"),
    limit: int = typer.Option(None, "--limit", "-n", help="Run only first N prompts"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="OpenAI model to use"),
    registry: str = typer.Option(
        "src/shardguard/mcp_servers/mcp_registry.json",
        "--registry", "-r",
        help="Path to MCP JSON config",
    ),
):
    """Run the ShardGuard evaluation harness against live MCPs."""
    import sys as _sys
    from pathlib import Path

    repo_root = Path(__file__).parent.parent.parent  # ShardGuard/
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))

    from eval.run_eval import main as eval_main  # type: ignore

    args = ["eval"]
    if category:
        args += ["--category", category]
    if mode:
        args += ["--mode", mode]
    if limit:
        args += ["--limit", str(limit)]
    if model:
        args += ["--model", model]
    if registry:
        args += ["--registry", registry]

    _sys.argv = args
    asyncio.run(eval_main())


if __name__ == "__main__":
    app()
