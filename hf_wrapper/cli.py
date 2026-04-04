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
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from hf_wrapper.constants import COMMON_TAGS, PIPELINE_TAGS
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
    suggest_sharding,
    min_gpus_for_model,
    QuantOption,
    ShardingOption,
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
    per_gpu_vram_mb: int = 0,
    max_gpus: int = 1,
) -> tuple[str, str]:
    """Return (status_text, rich_style) for a model's fit status.

    When *max_gpus* > 1 and the model doesn't fit in a single GPU, the cell
    will show the minimum GPU count required for sharding rather than a plain
    red "too large" label.
    """
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

    # Doesn't fit on a single GPU even with quantization.
    # If we have multi-GPU info, try sharding.
    if max_gpus > 1 and per_gpu_vram_mb > 0:
        shard_opts = suggest_sharding(param_count, per_gpu_vram_mb, max_gpus=max_gpus)
        for opt in shard_opts:
            if opt.gpu_count == 1:
                continue  # already checked above
            if opt.native_fits:
                return f"↕ {opt.gpu_count}× GPU (native)", "cyan"
            if opt.best_quant:
                tag = opt.best_quant.level.key.upper()
                return f"↕ {opt.gpu_count}× GPU (~{tag})", "bright_cyan"

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

    if hw.homogeneous_gpu_count > 1:
        per_gpu_mb = hw.best_gpu_vram_mb
        total_mb = hw.homogeneous_gpu_count * per_gpu_mb
        out.print(
            f"[bold cyan]Multi-GPU config[/bold cyan]: "
            f"[bold]{hw.homogeneous_gpu_count}×[/bold] {hw.gpus[0].name}  "
            f"([bold]{_fmt_mb(per_gpu_mb)}[/bold] each, "
            f"[bold]{_fmt_mb(total_mb)}[/bold] total tensor-parallel budget)\n"
            f"[dim]Use [bold]hfw search <query> --gpus {hw.homogeneous_gpu_count}[/bold] "
            f"to include shardable models in search results.[/dim]"
        )
        out.print()


# ---------------------------------------------------------------------------
# Shell-completion helpers
# ---------------------------------------------------------------------------

def _complete_task(ctx, param, incomplete):
    from click.shell_completion import CompletionItem
    lower = incomplete.lower()
    return [CompletionItem(t) for t in PIPELINE_TAGS if t.startswith(lower)]


def _complete_tag(ctx, param, incomplete):
    from click.shell_completion import CompletionItem
    lower = incomplete.lower()
    prefix  = [t for t in COMMON_TAGS if t.startswith(lower)]
    substr  = [t for t in COMMON_TAGS if lower in t and not t.startswith(lower)]
    return [CompletionItem(t) for t in prefix + substr]


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
    type=click.Choice(PIPELINE_TAGS, case_sensitive=False),
    shell_complete=_complete_task,
    help="Filter by pipeline task.",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    shell_complete=_complete_tag,
    help="Filter by Hub tag (repeatable: --tag pytorch --tag en).",
)
@click.option(
    "--download",
    "download_index",
    default=None,
    type=int,
    metavar="N",
    help="Download result number N without interactive prompt (1-based).",
)
@click.option(
    "--gpus",
    default=None,
    type=int,
    metavar="N",
    help=(
        "Maximum number of GPUs per node to consider for sharding "
        "(default: detected GPU count, min 1)."
    ),
)
def cmd_search(
    query: str,
    limit: int,
    show_all: bool,
    task: Optional[str],
    tags: tuple,
    download_index: Optional[int],
    gpus: Optional[int],
) -> None:
    """
    Search HuggingFace Hub for QUERY, showing only models compatible with
    your hardware.  Models that need quantization to fit are marked with the
    recommended quant level.  Models that require sharding across multiple
    GPUs are shown with cyan ↕ markers.

    After results are shown you will be prompted to download a model by
    entering its result number.  Pass --download N to skip the prompt.
    """
    from huggingface_hub import HfApi

    hw = get_hw()
    available_mb = hw.effective_memory_mb
    per_gpu_vram_mb = hw.best_gpu_vram_mb
    max_gpus = gpus if gpus is not None else hw.homogeneous_gpu_count
    max_gpus = max(max_gpus, 1)

    gpu_note = ""
    if max_gpus > 1:
        gpu_note = f"  [dim]({max_gpus}× GPU sharding on)[/dim]"

    err.print(f"\n[bold cyan]Hardware[/bold cyan]: {hw.inference_device}  "
              f"([bold]{_fmt_mb(available_mb)}[/bold] available){gpu_note}")
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
        if tags:
            kwargs["filter"] = list(tags)
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
    table.add_column("#",      justify="right",  min_width=3, style="dim")
    table.add_column("Model",  style="cyan",     no_wrap=True, min_width=30)
    table.add_column("Params", justify="right",  min_width=7)
    table.add_column("Dtype",  justify="center", min_width=6)
    table.add_column("Status / Recommendation", min_width=28)
    table.add_column("Downloads", justify="right", min_width=10)

    shown_models: List[str] = []   # repo_ids in display order (1-based index)

    for info in models:
        mem = estimate_from_listing(info)
        fits = mem.fits_in(available_mb)
        can_quant = mem.param_count and can_fit_with_any_quant(mem.param_count, available_mb)
        # Can it fit via sharding across available GPUs?
        can_shard = False
        if not fits and not can_quant and mem.param_count and max_gpus > 1:
            shard_opts = suggest_sharding(mem.param_count, per_gpu_vram_mb, max_gpus=max_gpus)
            can_shard = any(o.is_viable for o in shard_opts if o.gpu_count > 1)
        size_known = mem.param_count is not None

        if not show_all and not fits and not can_quant and not can_shard and size_known:
            continue

        shown_models.append(info.id or "")
        idx = len(shown_models)

        status_text, status_style = _status_cell(
            mem, available_mb,
            per_gpu_vram_mb=per_gpu_vram_mb,
            max_gpus=max_gpus,
        )

        downloads = getattr(info, "downloads", None)
        dl_str = f"{downloads:,}" if downloads else "—"

        table.add_row(
            str(idx),
            info.id or "—",
            mem.param_str,
            mem.dtype,
            Text(status_text, style=status_style),
            Text(dl_str, style="dim"),
        )

    if not shown_models:
        out.print(
            "[yellow]No compatible models found. "
            "Try --show-all to see all results.[/yellow]"
        )
        return

    out.print(table)
    legend_parts = [
        "Green = fits natively",
        "Yellow = fits with quantization",
        "Cyan ↕ = requires multi-GPU sharding",
        "Red = too large",
    ]
    out.print(
        f"\n[dim]Showed {len(shown_models)}/{len(models)} models. "
        + "  ".join(legend_parts)
        + "[/dim]"
    )
    if not show_all:
        hidden = len(models) - len(shown_models)
        if hidden:
            out.print(
                f"[dim]{hidden} model(s) hidden (too large even for {max_gpus}× GPU). "
                f"Use --show-all to see them.[/dim]"
            )

    # ------------------------------------------------------------------
    # Download prompt
    # ------------------------------------------------------------------
    chosen_id: Optional[str] = None

    if download_index is not None:
        # Non-interactive: --download N flag
        if 1 <= download_index <= len(shown_models):
            chosen_id = shown_models[download_index - 1]
        else:
            err.print(
                f"[red]--download {download_index} is out of range "
                f"(1–{len(shown_models)}).[/red]"
            )
            sys.exit(1)
    elif sys.stdin.isatty():
        # Interactive prompt
        try:
            raw = input(
                f"\nDownload a model? Enter number (1–{len(shown_models)}) "
                "or press Enter to skip: "
            ).strip()
            if raw:
                n = int(raw)
                if 1 <= n <= len(shown_models):
                    chosen_id = shown_models[n - 1]
                else:
                    err.print(
                        f"[red]{n} is out of range (1–{len(shown_models)}).[/red]"
                    )
                    sys.exit(1)
        except ValueError:
            err.print("[red]Invalid input — expected a number.[/red]")
            sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            out.print("\n[dim]Download skipped.[/dim]")
            return

    if chosen_id:
        download_root = _get_download_root()
        local_dir = download_root / chosen_id
        _run_download_with_check(chosen_id, ["--local-dir", str(local_dir)])


def _get_download_root() -> Path:
    """Return the model download root directory."""
    shared_root = os.environ.get("AI_FLUX_SHARED_ROOT")
    if shared_root:
        return Path(shared_root) / "models"
    return Path("/share/hpc_shared/models")


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
    Downloads to $AI_FLUX_SHARED_ROOT/models/<repo> or /share/hpc_shared/models/<repo>
    unless --local-dir is passed explicitly.
    """
    # If caller passed --local-dir themselves, respect it and skip our routing
    extra = list(extra_args)
    if "--local-dir" in extra:
        # Let _download_model handle the check and forward; but use provided path
        _run_download_with_check(repo_id, extra)
        return

    # Use shared model root, then forward extra args on top
    download_root = _get_download_root()
    local_dir = download_root / repo_id
    _run_download_with_check(repo_id, ["--local-dir", str(local_dir)] + extra)


def _run_download_with_check(repo_id: str, extra_args: List[str]) -> None:
    """Hardware-check then forward to hub download with the given extra args."""
    hw = get_hw()
    available_mb = hw.effective_memory_mb

    err.print(
        f"\n[bold cyan]Hardware[/bold cyan]: {hw.inference_device}  "
        f"([bold]{_fmt_mb(available_mb)}[/bold] available)\n"
    )

    # Show which directory we're downloading to, if any
    if "--local-dir" in extra_args:
        idx = extra_args.index("--local-dir")
        if idx + 1 < len(extra_args):
            err.print(f"[dim]Download destination:[/dim] [bold]{extra_args[idx + 1]}[/bold]\n")

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
        param_count = mem.param_count
        native_mb = mem.native_vram_mb or 0
        err.print(
            f"[bold red]✗ {repo_id}[/bold red] requires "
            f"[bold]{_fmt_mb(native_mb)}[/bold] at {mem.dtype}, "
            f"but you have [bold]{_fmt_mb(available_mb)}[/bold] available."
        )
        if can_fit_with_any_quant(param_count, available_mb):
            options = suggest_quantizations(param_count, available_mb)
            err.print("\n[bold yellow]The model can fit with quantization:[/bold yellow]")
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
            # Single-GPU quant won't help — show multi-GPU sharding options
            per_gpu_mb = hw.best_gpu_vram_mb
            gpu_name = hw.gpus[0].name if hw.gpus else "GPU"
            if per_gpu_mb > 0:
                shard_opts = suggest_sharding(param_count, per_gpu_mb, max_gpus=8)
                viable = [o for o in shard_opts if o.is_viable and o.gpu_count > 1]
                if viable:
                    err.print(
                        "\n[bold cyan]Single-GPU options exhausted. "
                        "Multi-GPU (tensor-parallel) sharding options:[/bold cyan]"
                    )
                    _print_sharding_table(shard_opts, gpu_name=gpu_name)
                    min_opt = viable[0]
                    tp_flag = f"--tensor-parallel-size {min_opt.gpu_count}"
                    err.print(
                        f"\n[bold]Minimum config[/bold]: "
                        f"[bold cyan]{min_opt.gpu_count}× {gpu_name}[/bold cyan] "
                        f"([bold]{_fmt_mb(min_opt.total_vram_mb)}[/bold] total)\n"
                        f"Launch with vLLM: [cyan]vllm serve {repo_id} {tp_flag}[/cyan]\n"
                    )
                else:
                    err.print(
                        "[bold red]This model cannot fit in your hardware memory even "
                        "with the most aggressive quantization (Q2) or sharding across "
                        "up to 8 GPUs of this type.[/bold red]\n"
                        "Consider:\n"
                        "  • A smaller model variant (fewer parameters)\n"
                        "  • A node with more / larger GPUs\n"
                        "  • Cloud-based inference\n"
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

    _forward_to_hub_cli(["download", repo_id] + extra_args)



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


def _print_sharding_table(
    options: List[ShardingOption],
    gpu_name: str = "GPU",
) -> None:
    """Print a tensor-parallel scaling table to stderr."""
    table = Table(
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("GPUs (TP)", justify="center", min_width=10)
    table.add_column("Total VRAM", justify="right", min_width=11)
    table.add_column("Fits natively?", justify="center", min_width=14)
    table.add_column("Best option", min_width=28)

    for opt in options:
        if opt.native_fits:
            fits_cell = Text("✓ yes", style="green")
            best_cell = Text("BF16 / FP16 (native)", style="green")
        elif opt.best_quant:
            fits_cell = Text("~ quant", style="yellow")
            label = opt.best_quant.level.display
            vram = _fmt_mb(opt.best_quant.estimated_vram_mb)
            best_cell = Text(f"{label}  (~{vram})", style=opt.best_quant.quality_color)
        else:
            fits_cell = Text("✗ no", style="red")
            best_cell = Text("too large even at Q2", style="dim")

        table.add_row(
            Text(f"{opt.gpu_count}× {gpu_name}", style="cyan"),
            _fmt_mb(opt.total_vram_mb),
            fits_cell,
            best_cell,
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
      hardware    Show detected GPU/VRAM and available memory
      search      Search models filtered to what fits your hardware
      completion  Print shell tab-completion setup instructions

    Enhanced commands:
      download   Hardware compatibility check before downloading

    \b
    All other HuggingFace Hub commands (auth, cache, upload, models, …)
    are forwarded transparently to the original hf CLI.
    Run `hf <command> --help` for details on any hub command.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# `hf completion`
# ---------------------------------------------------------------------------

@click.command("completion")
@click.argument(
    "shell",
    type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
    default=None,
    required=False,
)
def cmd_completion(shell: Optional[str]) -> None:
    """
    Print shell tab-completion setup instructions.

    SHELL is auto-detected from $SHELL if not specified.
    Supported: bash, zsh, fish.

    \b
    Quick setup:
      eval "$(hfw completion)"       # auto-detect shell
      eval "$(hfw completion zsh)"   # explicit shell
      eval "$(hfw completion bash)"
    """
    if shell is None:
        shell_env = os.environ.get("SHELL", "")
        shell_name = os.path.basename(shell_env).lower()
        if shell_name in ("bash", "zsh", "fish"):
            shell = shell_name
        else:
            err.print(
                f"[yellow]Could not detect shell from $SHELL ('{shell_env}'). "
                "Pass it explicitly: hfw completion [bash|zsh|fish][/yellow]"
            )
            sys.exit(1)

    shell = shell.lower()
    prog = "hfw"  # use the non-shadowing alias
    env_var = f"_{prog.upper()}_COMPLETE"

    if shell == "zsh":
        out.print(
            f'# Add to ~/.zshrc — enables tab-completion for {prog}:\n'
            f'eval "$({env_var}=zsh_source {prog})"'
        )
    elif shell == "bash":
        out.print(
            f'# Add to ~/.bashrc — enables tab-completion for {prog}:\n'
            f'eval "$({env_var}=bash_source {prog})"'
        )
    elif shell == "fish":
        out.print(
            f'# Run once to install fish completions for {prog}:\n'
            f'{env_var}=fish_source {prog} '
            f'> ~/.config/fish/completions/{prog}.fish'
        )


cli.add_command(cmd_hardware)
cli.add_command(cmd_search)
cli.add_command(cmd_download)
cli.add_command(cmd_completion)


def main() -> None:
    cli(standalone_mode=False)
    # standalone_mode=False means Click won't call sys.exit itself; we do it
    # only when needed.  This keeps the import path clean for pass-through.


if __name__ == "__main__":
    main()
