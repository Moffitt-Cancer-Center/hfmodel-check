"""
cli.py — Hardware-aware wrapper for the HuggingFace Hub CLI.

Intercepts:
  hf search  <query>   — list models filtered by available GPU/RAM
  hf download <model>  — check hardware fit before downloading
  hf hardware          — show detected hardware summary

All other subcommands are forwarded to huggingface_hub's own CLI.
"""

from __future__ import annotations

import sys
import os
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from hf_wrapper.hardware import detect_hardware, HardwareInfo
from hf_wrapper.model_info import (
    get_model_memory_info,
    estimate_from_listing,
    ModelMemoryInfo,
)
from hf_wrapper.quantization import (
    suggest_quantizations,
    best_quantization,
    fits_natively,
    can_fit_with_any_quant,
    QuantOption,
)

# Two consoles: status/warnings → stderr so stdout stays pipe-friendly
err = Console(stderr=True)
out = Console()

# Lazy-cached hardware detection
_hw: Optional[HardwareInfo] = None


def get_hw() -> HardwareInfo:
    global _hw
    if _hw is None:
        _hw = detect_hardware()
    return _hw


# ---------------------------------------------------------------------------
# Forwarding to the original huggingface_hub CLI
# ---------------------------------------------------------------------------

def _forward_to_hub_cli(args: List[str]) -> None:
    """
    Delegate to huggingface_hub's own CLI by calling its entry point directly.
    This avoids any PATH collision between our `hf` script and the original.
    Supports huggingface_hub ≥ 1.0 (typer-based) and older argparse-based builds.
    """
    original_argv = sys.argv[:]
    sys.argv = ["hf"] + list(args)
    try:
        # huggingface_hub ≥ 1.0 — typer-based CLI lives in huggingface_hub.cli.hf
        try:
            from huggingface_hub.cli.hf import main as _hub_main  # type: ignore[import]
            _hub_main()
            return
        except ImportError:
            pass
        # huggingface_hub 0.20–0.x — single _cli module
        try:
            from huggingface_hub._cli import main as _hub_main  # type: ignore[import]
            _hub_main()
            return
        except ImportError:
            pass
        # Older fallback — commands sub-package
        from huggingface_hub.commands.huggingface_cli import main as _hub_main  # type: ignore[import]
        _hub_main()
    except SystemExit:
        raise
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_mb(mb: int) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


def _status_cell(
    mem: ModelMemoryInfo,
    available_mb: int,
) -> tuple[str, str]:
    """Return (status_text, rich_style) for a model's fit status."""
    if mem.native_vram_mb is None or mem.param_count is None:
        return "? unknown size", "dim"

    param_count: int = mem.param_count  # narrowed

    if mem.fits_in(available_mb):
        return f"✓ fits  ({_fmt_mb(mem.native_vram_mb)})", "green"

    if can_fit_with_any_quant(param_count, available_mb):
        best = best_quantization(param_count, available_mb)
        if best:
            label = best.level.key.upper()
            return f"~ {label} ({_fmt_mb(best.estimated_vram_mb)})", "yellow"

    return f"✗ too large ({_fmt_mb(mem.native_vram_mb)})", "red"


# ---------------------------------------------------------------------------
# `hf hardware`
# ---------------------------------------------------------------------------

@click.command("hardware")
def cmd_hardware() -> None:
    """Show detected hardware and available memory for inference."""
    hw = get_hw()
    out.print("\n[bold cyan]Detected Hardware[/bold cyan]")
    out.print(hw.summary())
    out.print()


# ---------------------------------------------------------------------------
# `hf search`
# ---------------------------------------------------------------------------

@click.command("search")
@click.argument("query")
@click.option(
    "--limit",
    default=20,
    show_default=True,
    help="Maximum number of models to fetch from the Hub.",
)
@click.option(
    "--show-all",
    is_flag=True,
    default=False,
    help="Include models that cannot fit even with maximum quantization.",
)
@click.option(
    "--task",
    default=None,
    help="Filter by pipeline tag (e.g. text-generation, fill-mask).",
)
def cmd_search(query: str, limit: int, show_all: bool, task: Optional[str]) -> None:
    """
    Search HuggingFace Hub for QUERY, showing only models compatible with
    your hardware.  Models that need quantization to fit are marked with the
    recommended quant level.
    """
    from huggingface_hub import HfApi

    hw = get_hw()
    available_mb = hw.effective_memory_mb

    err.print(f"\n[bold cyan]Hardware[/bold cyan]: {hw.inference_device}  "
              f"([bold]{_fmt_mb(available_mb)}[/bold] available)")
    err.print(f"[dim]Searching for models matching '[bold]{query}[/bold]'…[/dim]\n")

    api = HfApi()

    with err.status("Fetching model list from HuggingFace Hub…"):
        kwargs: dict = dict(
            search=query,
            limit=limit,
            sort="downloads",
            full=True,
            fetch_config=False,
        )
        if task:
            kwargs["pipeline_tag"] = task
        try:
            models = list(api.list_models(**kwargs))
        except Exception as exc:
            msg = str(exc)
            if "SSL" in msg or "CERTIFICATE" in msg:
                err.print(
                    "[red]TLS/SSL error connecting to HuggingFace Hub.[/red]\n"
                    "If you are behind a corporate proxy with SSL inspection, "
                    "set the [bold]REQUESTS_CA_BUNDLE[/bold] or "
                    "[bold]CURL_CA_BUNDLE[/bold] environment variable to your "
                    "CA bundle path, or set "
                    "[bold]HF_HUB_DISABLE_SSL_VERIFICATION=1[/bold] (insecure)."
                )
            else:
                err.print(f"[red]Error fetching models: {exc}[/red]")
            sys.exit(1)

    if not models:
        out.print("[yellow]No models found for that query.[/yellow]")
        return

    table = Table(
        show_header=True,
        header_style="bold magenta",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Model", style="cyan", no_wrap=True, min_width=30)
    table.add_column("Params", justify="right", min_width=7)
    table.add_column("Dtype",  justify="center", min_width=6)
    table.add_column("Status / Recommendation", min_width=28)
    table.add_column("Downloads", justify="right", min_width=10)

    shown = 0
    for info in models:
        mem = estimate_from_listing(info)
        fits = mem.fits_in(available_mb)
        can_quant = mem.param_count and can_fit_with_any_quant(mem.param_count, available_mb)
        size_known = mem.param_count is not None

        if not show_all and not fits and not can_quant and size_known:
            continue  # skip models that can't fit at all

        status_text, status_style = _status_cell(mem, available_mb)

        downloads = getattr(info, "downloads", None)
        dl_str = f"{downloads:,}" if downloads else "—"

        table.add_row(
            info.id or "—",
            mem.param_str,
            mem.dtype,
            Text(status_text, style=status_style),
            Text(dl_str, style="dim"),
        )
        shown += 1

    if shown == 0:
        out.print(
            "[yellow]No compatible models found. "
            "Try --show-all to see all results.[/yellow]"
        )
        return

    out.print(table)
    out.print(
        f"\n[dim]Showed {shown}/{len(models)} models. "
        f"Green = fits natively, Yellow = fits with quantization, "
        f"Red = too large even with Q2.[/dim]"
    )
    if not show_all:
        hidden = len(models) - shown
        if hidden:
            out.print(
                f"[dim]{hidden} model(s) hidden (too large). Use --show-all to see them.[/dim]"
            )


# ---------------------------------------------------------------------------
# `hf download` — intercept to run a hardware check first
# ---------------------------------------------------------------------------

@click.command(
    "download",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("repo_id")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def cmd_download(repo_id: str, extra_args: tuple) -> None:
    """
    Download a repo (wraps `hf download`), with a hardware compatibility check.
    """
    hw = get_hw()
    available_mb = hw.effective_memory_mb

    err.print(
        f"\n[bold cyan]Hardware[/bold cyan]: {hw.inference_device}  "
        f"([bold]{_fmt_mb(available_mb)}[/bold] available)\n"
    )

    with err.status(f"Checking model [bold]{repo_id}[/bold]…"):
        try:
            mem = get_model_memory_info(repo_id)
        except Exception:
            mem = None

    if mem is None or mem.param_count is None:
        err.print(
            f"[yellow]Could not determine model size for [bold]{repo_id}[/bold]. "
            "Proceeding with download.[/yellow]\n"
        )
    elif mem.fits_in(available_mb):
        native_mb = mem.native_vram_mb or 0
        err.print(
            f"[green]✓ [bold]{repo_id}[/bold] fits in your "
            f"{_fmt_mb(available_mb)} ({_fmt_mb(native_mb)} needed at "
            f"{mem.dtype}).[/green]\n"
        )
    else:
        param_count = mem.param_count  # narrowed: not None because fits_in returned False with a value
        native_mb = mem.native_vram_mb or 0
        # Model does NOT fit natively
        err.print(
            f"[bold red]✗ {repo_id}[/bold red] requires "
            f"[bold]{_fmt_mb(native_mb)}[/bold] at {mem.dtype}, "
            f"but you have [bold]{_fmt_mb(available_mb)}[/bold] available."
        )

        if can_fit_with_any_quant(param_count, available_mb):
            options = suggest_quantizations(param_count, available_mb)
            err.print(
                "\n[bold yellow]The model can fit with quantization:[/bold yellow]"
            )
            _print_quant_table(options)

            best = options[0]
            err.print(
                f"\n[bold]Recommendation[/bold]: "
                f"use a [bold yellow]{best.level.display}[/bold yellow] variant "
                f"(~[bold]{_fmt_mb(best.estimated_vram_mb)}[/bold]).\n"
                f"Search the Hub for pre-quantized versions:\n"
                f"  [cyan]hf search \"{repo_id.split('/')[-1]}\" --task text-generation[/cyan]\n"
                f"Or convert locally with [cyan]llama.cpp[/cyan] / [cyan]ollama[/cyan].\n"
            )
        else:
            err.print(
                "[bold red]This model cannot fit in your hardware memory even "
                "with the most aggressive quantization (Q2).[/bold red]\n"
                "Consider:\n"
                "  • A smaller model variant (fewer parameters)\n"
                "  • Running inference on CPU (slow but functional)\n"
                "  • Cloud-based inference\n"
            )

        if not _confirm_proceed():
            err.print("[dim]Download cancelled.[/dim]")
            sys.exit(0)

    # Forward to the real hub CLI
    _forward_to_hub_cli(["download", repo_id] + list(extra_args))


def _print_quant_table(options: List[QuantOption]) -> None:
    table = Table(
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("Quantization",  min_width=18)
    table.add_column("Est. VRAM",  justify="right", min_width=10)
    table.add_column("Quality Loss",  min_width=10)
    table.add_column("Notes", min_width=35)

    for opt in options:
        color = opt.quality_color
        rec = " ★" if opt.is_recommended and opt is options[0] else ""
        table.add_row(
            Text(opt.level.display + rec, style=color),
            _fmt_mb(opt.estimated_vram_mb),
            Text(opt.level.quality_tag, style=color),
            Text(opt.level.notes, style="dim"),
        )
    err.print(table)


def _confirm_proceed() -> bool:
    """Ask user if they want to download anyway. Returns True to proceed."""
    try:
        answer = input(
            "Download anyway? This model may not run on your hardware. [y/N] "
        ).strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ---------------------------------------------------------------------------
# Root group — routes to our commands or forwards to the real hf
# ---------------------------------------------------------------------------

class _PassthroughGroup(click.Group):
    """
    Click Group that forwards any unrecognised *command* to the real
    huggingface_hub CLI.  Flags (starting with `-`) are handled normally by
    Click so that `hf --help` still shows our own help text.
    """

    def parse_args(self, ctx: click.Context, args: List[str]) -> List[str]:
        # Only forward if the first token looks like a command name (no dash prefix)
        # that is NOT one of our own registered commands.
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            _forward_to_hub_cli(args)
            sys.exit(0)
        return super().parse_args(ctx, args)


@click.group(
    cls=_PassthroughGroup,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    Hardware-aware wrapper for the HuggingFace Hub CLI.

    \b
    New commands (hardware-aware):
      hardware   Show detected GPU/VRAM and available memory
      search     Search models filtered to what fits your hardware

    Enhanced commands:
      download   Hardware compatibility check before downloading

    \b
    All other HuggingFace Hub commands (auth, cache, upload, models, …)
    are forwarded transparently to the original hf CLI.
    Run `hf <command> --help` for details on any hub command.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(cmd_hardware)
cli.add_command(cmd_search)
cli.add_command(cmd_download)


def main() -> None:
    cli(standalone_mode=False)
    # standalone_mode=False means Click won't call sys.exit itself; we do it
    # only when needed.  This keeps the import path clean for pass-through.


if __name__ == "__main__":
    main()
