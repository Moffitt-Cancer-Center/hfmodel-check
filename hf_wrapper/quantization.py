"""
quantization.py — Determine which quantizations fit in available GPU memory.
"""

from __future__ import annotations

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
