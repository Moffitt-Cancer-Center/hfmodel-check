"""
quantization.py — Determine which quantizations fit in available GPU memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from hf_wrapper.model_info import DTYPE_BYTES, INFERENCE_OVERHEAD


# ---------------------------------------------------------------------------
# Catalogue of quantization levels, ordered best→worst quality
# ---------------------------------------------------------------------------

@dataclass
class QuantLevel:
    key: str           # internal key matching DTYPE_BYTES
    display: str       # human-readable name
    bytes_per_param: float
    quality_tag: str   # "none" | "minimal" | "low" | "moderate" | "high" | "extreme"
    notes: str = ""    # brief hint for the user


QUANT_LEVELS: List[QuantLevel] = [
    QuantLevel("bf16",    "BF16 (native)",    2.000, "none",
               "Full quality — native half-precision"),
    QuantLevel("fp16",    "FP16 (native)",    2.000, "none",
               "Full quality — native half-precision"),
    QuantLevel("int8",    "INT8 / Q8_0",      1.000, "minimal",
               "Nearly lossless; 2× VRAM savings"),
    QuantLevel("q6_k",    "Q6_K  (GGUF)",     0.750, "minimal",
               "Minimal loss; good balance"),
    QuantLevel("q5_k_m",  "Q5_K_M (GGUF)",   0.625, "low",
               "Recommended 5-bit; excellent quality"),
    QuantLevel("q4_k_m",  "Q4_K_M (GGUF)",   0.500, "low",
               "Most popular; solid quality/size trade-off"),
    QuantLevel("q4_0",    "Q4_0  (GGUF)",     0.500, "moderate",
               "Older 4-bit; slightly more loss than Q4_K_M"),
    QuantLevel("q3_k_m",  "Q3_K_M (GGUF)",   0.375, "high",
               "Noticeable quality drop; last resort"),
    QuantLevel("q2_k",    "Q2_K  (GGUF)",     0.250, "extreme",
               "Severe quality loss; emergency use only"),
]

# Map key → level for quick lookup
QUANT_BY_KEY = {q.key: q for q in QUANT_LEVELS}


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class QuantOption:
    level: QuantLevel
    estimated_vram_mb: int

    @property
    def is_recommended(self) -> bool:
        return self.level.quality_tag in ("none", "minimal", "low")

    @property
    def quality_color(self) -> str:
        """Rich markup color based on quality loss."""
        return {
            "none":     "green",
            "minimal":  "green",
            "low":      "bright_green",
            "moderate": "yellow",
            "high":     "red",
            "extreme":  "bright_red",
        }.get(self.level.quality_tag, "white")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def suggest_quantizations(
    param_count: int,
    available_mb: int,
) -> List[QuantOption]:
    """
    Return all quantization levels (best first) whose estimated VRAM fits
    within *available_mb*.

    Usage::

        options = suggest_quantizations(7_000_000_000, 8192)
        best = options[0]  # highest quality that fits
    """
    options: List[QuantOption] = []
    for level in QUANT_LEVELS:
        vram = int(param_count * level.bytes_per_param * INFERENCE_OVERHEAD / (1024 * 1024))
        if vram <= available_mb:
            options.append(QuantOption(level=level, estimated_vram_mb=vram))
    return options


def best_quantization(
    param_count: int,
    available_mb: int,
) -> Optional[QuantOption]:
    """Return the highest-quality quantization that fits, or None."""
    opts = suggest_quantizations(param_count, available_mb)
    return opts[0] if opts else None


def native_vram_mb(param_count: int, dtype: str = "bf16") -> int:
    """Estimate inference VRAM at the model's native dtype."""
    bpp = DTYPE_BYTES.get(dtype, 2.0)
    return int(param_count * bpp * INFERENCE_OVERHEAD / (1024 * 1024))


def fits_natively(param_count: int, available_mb: int, dtype: str = "bf16") -> bool:
    return native_vram_mb(param_count, dtype) <= available_mb


def can_fit_with_any_quant(param_count: int, available_mb: int) -> bool:
    """True if even the most aggressive quantization (Q2) would fit."""
    most_aggressive = QUANT_LEVELS[-1]
    min_vram = int(param_count * most_aggressive.bytes_per_param * INFERENCE_OVERHEAD / (1024 * 1024))
    return min_vram <= available_mb


# ---------------------------------------------------------------------------
# Multi-GPU / tensor-parallel sharding
# ---------------------------------------------------------------------------

@dataclass
class ShardingOption:
    """Describes how a model can be served across N GPUs on the same node."""
    gpu_count: int               # tensor-parallel degree (1, 2, 4, 8, …)
    per_gpu_vram_mb: int         # VRAM available on each GPU
    total_vram_mb: int           # gpu_count × per_gpu_vram_mb
    native_fits: bool            # model at native dtype fits within total_vram
    best_quant: Optional[QuantOption]  # best quant if native doesn't fit, else None

    @property
    def is_viable(self) -> bool:
        """True if the model can run at this GPU count (natively or quantized)."""
        return self.native_fits or self.best_quant is not None

    @property
    def status_label(self) -> tuple[str, str]:
        """Return (text, rich_style) for display."""
        if self.native_fits:
            return f"{self.gpu_count}× GPU  (native)", "green"
        if self.best_quant:
            tag = self.best_quant.level.key.upper()
            return f"{self.gpu_count}× GPU  (~{tag})", "yellow"
        return f"{self.gpu_count}× GPU  too large", "red"


_TP_STEPS = (1, 2, 4, 8, 16)   # standard tensor-parallel sizes


def suggest_sharding(
    param_count: int,
    per_gpu_vram_mb: int,
    max_gpus: int = 8,
) -> List[ShardingOption]:
    """
    Return sharding options for increasing GPU counts (1, 2, 4, 8, …) up to
    *max_gpus*, describing whether the model fits at each scale.

    Tensor parallelism splits the model weight across GPUs so the effective
    VRAM budget is ``gpu_count × per_gpu_vram_mb``.

    Example::

        opts = suggest_sharding(70_000_000_000, per_gpu_vram_mb=40_960, max_gpus=8)
        for o in opts:
            print(o.gpu_count, o.native_fits, o.best_quant)
    """
    options: List[ShardingOption] = []
    for n in _TP_STEPS:
        if n > max_gpus:
            break
        total_mb = n * per_gpu_vram_mb
        nat_fits = native_vram_mb(param_count) <= total_mb
        best = None if nat_fits else best_quantization(param_count, total_mb)
        options.append(
            ShardingOption(
                gpu_count=n,
                per_gpu_vram_mb=per_gpu_vram_mb,
                total_vram_mb=total_mb,
                native_fits=nat_fits,
                best_quant=best,
            )
        )
        if nat_fits:
            # No point showing larger configs once it already fits natively.
            break
    return options


def min_gpus_for_model(
    param_count: int,
    per_gpu_vram_mb: int,
    dtype: str = "bf16",
) -> Optional[int]:
    """
    Return the minimum number of homogeneous GPUs needed to run the model at
    *dtype* natively, or ``None`` if even 16 GPUs wouldn't be enough.
    """
    needed_mb = native_vram_mb(param_count, dtype)
    if per_gpu_vram_mb <= 0:
        return None
    n = math.ceil(needed_mb / per_gpu_vram_mb)
    # Tensor parallelism must be a power-of-two (or TP=1)
    tp = 1
    while tp < n:
        tp *= 2
    return tp if tp <= 16 else None
