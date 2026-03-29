"""
model_info.py — Fetch HuggingFace model metadata and estimate memory requirements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from huggingface_hub import HfApi, ModelInfo
try:
    from huggingface_hub.errors import HfHubHTTPError  # huggingface_hub >= 0.23
except ImportError:
    from huggingface_hub.utils import HfHubHTTPError  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Bytes-per-parameter for common dtypes / quantization schemes
# ---------------------------------------------------------------------------

DTYPE_BYTES: Dict[str, float] = {
    # Full precision
    "fp32":     4.0,
    "float32":  4.0,
    # Half precision (native for most modern models)
    "fp16":     2.0,
    "float16":  2.0,
    "bf16":     2.0,
    "bfloat16": 2.0,
    # 8-bit
    "int8":     1.0,
    "q8_0":     1.0,
    "bnb8":     1.0,
    # 6-bit
    "q6_k":     0.75,
    # 5-bit
    "q5_k_m":   0.625,
    "q5_k_s":   0.625,
    "q5_0":     0.625,
    # 4-bit
    "q4_k_m":   0.5,
    "q4_k_s":   0.5,
    "q4_0":     0.5,
    "gptq":     0.5,
    "awq":      0.5,
    "bnb4":     0.5,
    "int4":     0.5,
    # 3-bit
    "q3_k_m":   0.375,
    "q3_k_s":   0.375,
    "q3_k_l":   0.375,
    # 2-bit
    "q2_k":     0.25,
    "q2_k_s":   0.25,
}

# Default overhead multiplier: KV cache, activations, framework buffers
INFERENCE_OVERHEAD = 1.20

# A model is "close to fitting" if any quantization down to Q4_K_M would fit.
# The ratio tells us: model needs up to N× available memory at native precision.
CLOSE_FIT_RATIO_THRESHOLD = 8.0  # bf16→q2 gives ~8× reduction


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ModelMemoryInfo:
    model_id: str
    param_count: Optional[int]   # total learnable parameters
    dtype: str                   # detected / assumed native dtype
    native_vram_mb: Optional[int]  # at native dtype with overhead

    # ----------------------------------------------------------------
    # Derived helpers
    # ----------------------------------------------------------------

    def vram_for_dtype(self, dtype: str) -> Optional[int]:
        """Return estimated VRAM (MB) needed at *dtype* with overhead."""
        if self.param_count is None:
            return None
        bpp = DTYPE_BYTES.get(dtype, 2.0)
        return int(self.param_count * bpp * INFERENCE_OVERHEAD / (1024 * 1024))

    def fits_in(self, available_mb: int) -> bool:
        if self.native_vram_mb is None:
            return True  # unknown — don't block
        return self.native_vram_mb <= available_mb

    def close_to_fitting(self, available_mb: int) -> bool:
        """True if a quantization exists that would make the model fit."""
        if self.param_count is None or available_mb == 0:
            return False
        # Minimum possible VRAM: q2 with overhead
        min_vram = int(self.param_count * 0.25 * INFERENCE_OVERHEAD / (1024 * 1024))
        return min_vram <= available_mb

    @property
    def param_str(self) -> str:
        if self.param_count is None:
            return "unknown"
        b = self.param_count / 1e9
        if b >= 1:
            return f"{b:.1f}B"
        m = self.param_count / 1e6
        return f"{m:.0f}M"


# ---------------------------------------------------------------------------
# Helpers: extract metadata from ModelInfo
# ---------------------------------------------------------------------------

def _param_count_from_safetensors(info: ModelInfo) -> Optional[int]:
    st = getattr(info, "safetensors", None)
    if st is None:
        return None
    # huggingface_hub>=0.20 exposes SafeTensorsInfo with .total
    total = getattr(st, "total", None)
    if isinstance(total, int) and total > 0:
        return total
    return None


_SIZE_RE = re.compile(
    r"(?<![a-zA-Z0-9])(\d+(?:\.\d+)?)\s*([bBmM])(?![a-zA-Z0-9])"
)


def _param_count_from_tags(info: ModelInfo) -> Optional[int]:
    """Parse parameter count from model tags and model id (e.g. '7b', '70B')."""
    candidates: List[str] = []
    if info.tags:
        candidates.extend(info.tags)
    if info.id:
        candidates.append(info.id)
    model_id_attr = getattr(info, "modelId", None)
    if model_id_attr:
        candidates.append(str(model_id_attr))

    for text in candidates:
        for m in _SIZE_RE.finditer(text):
            try:
                value = float(m.group(1))
                unit = m.group(2).lower()
                if unit == "b" and value <= 1000:  # sanity: 1000B params max
                    return int(value * 1_000_000_000)
                if unit == "m" and value <= 100_000:
                    return int(value * 1_000_000)
            except ValueError:
                pass
    return None


def _dtype_from_tags(info: ModelInfo) -> str:
    tags_lower = [t.lower() for t in (info.tags or [])]
    model_id_lower = (info.id or "").lower()

    checks = [
        ("gguf",   "gguf"),   # GGUF files encode their own quant
        ("gptq",   "gptq"),
        ("awq",    "awq"),
        ("bnb",    "bnb4"),   # bitsandbytes 4-bit
        ("int8",   "int8"),
        ("int4",   "int4"),
        ("fp16",   "fp16"),
        ("bf16",   "bf16"),
    ]
    all_text = tags_lower + [model_id_lower]
    for keyword, dtype in checks:
        if any(keyword in t for t in all_text):
            return dtype

    return "bf16"  # safe default for modern HF models


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_memory_info(
    model_id: str,
    api: Optional[HfApi] = None,
) -> Optional[ModelMemoryInfo]:
    """
    Fetch model metadata from the Hub and return a ModelMemoryInfo.
    Returns None on network error.
    """
    if api is None:
        api = HfApi()
    try:
        info = api.model_info(model_id, expand=["safetensors"])
    except (HfHubHTTPError, Exception):
        return None

    param_count = _param_count_from_safetensors(info)
    if param_count is None:
        param_count = _param_count_from_tags(info)

    dtype = _dtype_from_tags(info)

    native_vram_mb: Optional[int] = None
    if param_count is not None:
        bpp = DTYPE_BYTES.get(dtype, 2.0)
        native_vram_mb = int(
            param_count * bpp * INFERENCE_OVERHEAD / (1024 * 1024)
        )

    return ModelMemoryInfo(
        model_id=model_id,
        param_count=param_count,
        dtype=dtype,
        native_vram_mb=native_vram_mb,
    )


def estimate_from_listing(info: ModelInfo) -> ModelMemoryInfo:
    """
    Build a ModelMemoryInfo purely from a list_models() result (no extra API
    call).  Less accurate but fast enough for filtering long search results.
    """
    param_count = _param_count_from_safetensors(info)
    if param_count is None:
        param_count = _param_count_from_tags(info)

    dtype = _dtype_from_tags(info)

    native_vram_mb: Optional[int] = None
    if param_count is not None:
        bpp = DTYPE_BYTES.get(dtype, 2.0)
        native_vram_mb = int(
            param_count * bpp * INFERENCE_OVERHEAD / (1024 * 1024)
        )

    return ModelMemoryInfo(
        model_id=info.id or "",
        param_count=param_count,
        dtype=dtype,
        native_vram_mb=native_vram_mb,
    )
